import itertools
from contextlib import nullcontext
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import dynamo_timed, preserve_rng_state
from torch._guards import detect_fake_mode
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch._decomp.decompositions_for_rng import PhiloxStateTracker, rng_decompositions
from . import config
from .partitioners import default_partition
from ._aot_autograd.utils import (  # noqa: F401
from ._aot_autograd.logging_utils import (  # noqa: F401
from ._aot_autograd.functional_utils import (  # noqa: F401
from ._aot_autograd.schemas import (  # noqa: F401
from ._aot_autograd.subclass_utils import (  # noqa: F401
from ._aot_autograd.collect_metadata_analysis import (  # noqa: F401
from ._aot_autograd.input_output_analysis import (  # noqa: F401
from ._aot_autograd.traced_function_transforms import (  # noqa: F401
from ._aot_autograd.runtime_wrappers import (  # noqa: F401
from ._aot_autograd.dispatch_and_compile_graph import (  # noqa: F401
from ._aot_autograd.jit_compile_runtime_wrappers import (  # noqa: F401
def aot_export_module(mod: nn.Module, args, *, decompositions: Optional[Dict]=None, trace_joint: bool, output_loss_index: Optional[int]=None) -> Tuple[torch.fx.GraphModule, GraphSignature]:
    """
    This function takes in a module, and returns:
    (1) an FX graph that can be exported
    (2) some metadata about the graph

    If `trace_joint=True` we will return a joint graph of the forward + backward.

    The traced FX graph will have the following properties compared to the original module:
    (1) Inputs and outputs to the module will be pytree-flattened
    (2) Parameters and buffers on the module will be lifted into graph inputs,
        graph_inputs = (*parameters, *buffers, *user_inputs)
    (3) The graph will be fully functionalized
    (4) Any input mutations will be converted into additional outputs in the graph,
        meaning whoever calls this graph is responsible for applying the mutations
        back to the original inputs.
    (5) If is_joint is provided the graph will return parameter gradients in addition to user outputs.
        The graph output will look like:
        graph_outputs = (*updated_inputs, *user_outputs, *param_gradients)

    There are also several restrictions on what modules can use this API. In particular:
    (1) If trace_joint is specified, we expect the loss function to be **fused**
        into the module forward. One of the outputs to the forward must be a scalar loss,
        which is specified with `output_loss_index`.
        All other outputs to the forward are presumed to not require gradients.
    (2) This API cannot capture optimizers (although in theory we could build an API for this).
    (3) Metadata mutations on params/buffers/inputs are banned.
    (4) Data mutations on anything that requires gradients are banned (parameters)
    (5) If an input is mutated, it is not allowed to alias any other inputs.
    (6) Parameters must not be duplicated.
    """
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    params_and_buffers = {**dict(named_parameters), **dict(named_buffers)}
    params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
    params_and_buffers_flat = tuple(params_and_buffers_flat)
    params_len = len(params_and_buffers_flat)
    functional_call = create_functional_call(mod, params_spec, params_len)
    num_fw_outs = None
    if trace_joint:

        def fn_to_trace(*args):
            nonlocal num_fw_outs
            out = functional_call(*args)
            if output_loss_index is None:
                raise RuntimeError('If trace_joint=Trueit is required that one of your forward outputs must be a scalar loss.\nYou must specify the which (index) output is the loss with output_loss_index.')
            if isinstance(out, torch.Tensor):
                out = (out,)
            if not isinstance(out, (tuple, list)):
                raise RuntimeError(f'Expected forward output to be either a tensor or a list/tuple of tensors. found {type(out)}')
            for i, o in enumerate(out):
                if o.requires_grad and i != output_loss_index:
                    raise RuntimeError(f'Found an output of the forward that requires gradients, that was not the scalar loss.\nWe require all outputs to the forward that are not the scalar loss to not require gradient,\nbecause we will only compute a backward graph against the scalar loss.\nYou can fix this by calling .detach() on each of your forward outputs that is not the loss.\nYou specified that output index {output_loss_index} is the loss, but we found that\nthe output at index {i} requires gradients.')
            out_loss = out[output_loss_index]
            num_fw_outs = len(out)
            if not out_loss.requires_grad:
                raise RuntimeError(f'The output at index {output_loss_index} was marked as the loss, but it does not require gradients')
            if out_loss.numel() != 1:
                raise RuntimeError(f'We require the output marked as the loss (at index {output_loss_index}) to be a scalar, but it has shape {out_loss.shape}')
            return out
        ctx = nullcontext
    else:
        ctx = torch.no_grad
        fn_to_trace = functional_call
    full_args = []
    full_args.extend(params_and_buffers_flat)
    full_args.extend(args)
    with ctx():
        fx_g, metadata, in_spec, out_spec = _aot_export_function(fn_to_trace, full_args, decompositions=decompositions, num_params_buffers=params_len, no_tangents=True)
    if trace_joint:

        def flattened_joint(*args):
            fake_tangents = [None for _ in range(metadata.num_outputs + metadata.num_mutated_inp_runtime_indices)]
            fw_outs, gradients = fx_g(args, fake_tangents)
            assert len(gradients) == len(args)
            output_gradients = []
            for i, (a, grad) in enumerate(zip(args, gradients)):
                if isinstance(a, torch.Tensor) and a.requires_grad:
                    assert grad is not None, 'Found a parameter that did not receive a gradient.\n"This is most likely a bug, but if this needs to be supported please comment on this Github issue:\nhttps://github.com/pytorch/pytorch/issues/101192\n'
                    output_gradients.append(grad)
                else:
                    assert grad is None
            return (*fw_outs, *output_gradients)
        fx_g = make_fx(flattened_joint)(*full_args)
    user_args_flat = pytree.arg_tree_leaves(*args)
    return (fx_g, create_graph_signature(fx_g, metadata, in_spec, out_spec, user_args_flat=user_args_flat, params_and_buffers_flat=params_and_buffers_flat, param_names=list(named_parameters.keys()), buffer_names=list(named_buffers.keys()), trace_joint=trace_joint, num_user_fw_outs=num_fw_outs, loss_index=output_loss_index))