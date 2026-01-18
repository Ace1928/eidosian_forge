import warnings
from contextlib import nullcontext
from typing import Any, Callable, List, Tuple, Union
from unittest.mock import patch
import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import Tensor
from torch._decomp.decompositions_for_rng import PhiloxStateTracker
from torch._guards import detect_fake_mode
from torch._prims_common import CUDARngStateHelper
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.fx import Interpreter
from torch.fx.experimental.symbolic_shapes import definitely_false, sym_eq
from torch.nn.utils import stateless
from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import (
from .logging_utils import setup_stacktrace_preservation_hooks
from .schemas import (
from .subclass_utils import (
from .utils import maybe_to_fresh_input
def aot_dispatch_subclass(flat_fn_maybe_joint, args: List[Any], *, is_joint_structure: bool, meta: ViewAndMutationMeta, fw_only: Callable) -> SubclassTracingInfo:
    req_subclass_dispatch = requires_subclass_dispatch(args, meta)
    if not req_subclass_dispatch:
        return SubclassTracingInfo(plain_tensor_trace_fn=flat_fn_maybe_joint, plain_tensor_args=args, maybe_subclass_meta=None)
    subclass_meta = SubclassMeta()

    def inner_fn(fn, args, *, use_trace_joint: bool):
        all_args = wrap_tensor_subclasses_maybe_joint(args, is_joint_structure=use_trace_joint, meta=meta)
        wrapped_outs = fn(*all_args)
        if use_trace_joint:
            nonlocal subclass_meta
            assert isinstance(wrapped_outs, tuple) and len(wrapped_outs) == 2
            grad_inputs = wrapped_outs[1]
            subclass_meta.grad_input_metas = create_subclass_meta(grad_inputs)
        unwrapped_outs = unwrap_tensor_subclasses(wrapped_outs, is_joint_structure=use_trace_joint)
        return unwrapped_outs

    def joint_fn(primals, tangents):
        return inner_fn(flat_fn_maybe_joint, (primals, tangents), use_trace_joint=True)

    def fw_fn(*primals):
        return inner_fn(flat_fn_maybe_joint, primals, use_trace_joint=False)

    def metadata_fn(*primals):
        return inner_fn(fw_only, primals, use_trace_joint=False)
    args_unwrapped = unwrap_tensor_subclasses(args, is_joint_structure=is_joint_structure)
    if is_joint_structure:
        primals_unwrapped = args_unwrapped[0]
        fn_to_trace = joint_fn
    else:
        primals_unwrapped = args_unwrapped
        fn_to_trace = fw_fn
    meta_updated = run_functionalized_fw_and_collect_metadata(metadata_fn, keep_input_mutations=meta.keep_input_mutations, is_train=meta.is_train, requires_subclass_dispatch=True)(*primals_unwrapped)
    subclass_meta.fw_metadata = meta_updated
    return SubclassTracingInfo(plain_tensor_trace_fn=fn_to_trace, plain_tensor_args=args_unwrapped, maybe_subclass_meta=subclass_meta)