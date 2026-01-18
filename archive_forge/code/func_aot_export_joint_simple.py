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
def aot_export_joint_simple(func: Callable, args, *, trace_joint: bool, num_params_buffers: int=0, decompositions: Optional[Dict]=None) -> torch.fx.GraphModule:
    """
    A simplified version of export. Used by higher order operators.

    This function makes a high-level "no calling convention changes" guarantee:
    - If no inputs require grad (so we export an inference graph),
      there are *no* calling convention change between the exported graph, and "func".
    - If at least one input requires grad (so we trace out and export a joint fw-bw graph),
      Then if you were partition the graph into a separate forward and backward graph,
      The forward graph will have no calling convention changes compared to "func".

    The above also relies on some strong restrictions around which functions this API accepts:
    (1) `args` cannot contain any pytrees (they must have been pytree_flattened already)
    (2) `func` cannot mutate any inputs
    (3) The outputs of `func` cannot alias any inputs.

    Note: this function is only lightly tested today. It will probably be tested more heavily by higher order ops.
    """
    if trace_joint:
        ctx = nullcontext
    else:
        ctx = torch.no_grad
    with ctx():
        fx_g, metadata, in_spec, out_spec = _aot_export_function(func, args, decompositions=decompositions)
    if len([x for x in metadata.input_info if x.mutates_data or x.mutates_metadata]) != 0:
        raise RuntimeError(f'aot_export_joint_simple does not support input mutations. {str(metadata)}')
    if len([x for x in metadata.output_info if x.output_type != OutputType.non_alias]) != 0:
        raise RuntimeError(f'aot_export_joint_simple does not support outputs that alias inputs. {str(metadata)}')
    if type(in_spec) == pytree.LeafSpec:
        raise RuntimeError(f'aot_export_joint_simple requires inputs to be a single list/tuple. in_spec={str(in_spec)}')
    if len([x for x in in_spec.children_specs if type(x) != pytree.LeafSpec]) != 0:
        raise RuntimeError(f'aot_export_joint_simple requires individual inputs not to be pytrees. in_spec={str(in_spec)}')
    if type(out_spec) == pytree.LeafSpec:
        raise RuntimeError(f'aot_export_joint_simple requires outputs to be a single list/tuple. out_spec={str(out_spec)}')
    if len([x for x in out_spec.children_specs if type(x) != pytree.LeafSpec]) != 0:
        raise RuntimeError(f'aot_export_joint_simple requires individual outputs not to be pytrees. out_spec={str(out_spec)}')
    if config.debug_assert:
        fw_module, bw_module = aot_config.default_partition(fx_g, args, num_fwd_outputs=len(fw_metadata.output_infos))
        fake_mode = detect_fake_mode(args)
        if fake_mode is None:
            fake_mode = FakeTensorMode()
        with fake_mode:
            fw_module(*args)
    return fx_g