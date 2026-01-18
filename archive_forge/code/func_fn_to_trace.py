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