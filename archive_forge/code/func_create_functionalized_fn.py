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
def create_functionalized_fn(fn, args, *, meta: ViewAndMutationMeta, aot_config: AOTConfig, trace_joint: bool) -> Any:

    def _functionalized_f_helper(*args):
        f_args = pytree.tree_map(to_fun, args)
        disable_above = torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize))
        with disable_above, FunctionalTensorMode():
            f_outs = fn(*f_args)
        if trace_joint:
            assert isinstance(args, tuple) and len(args) == 2 and isinstance(args[0], (list, tuple))
            primals_before = args[0]
            primals_after = pytree.tree_map(from_fun, f_args[0])
            for f_inpt, before, after, inpt_info in zip(f_args[0], primals_before, primals_after, meta.input_info):
                if not inpt_info.mutates_metadata:
                    assert not has_metadata_mutation(f_inpt, before, check_only_storage_mutation=False), 'Found a graph input that had its metadata mutated in the backward. This is not supported'
                if has_data_mutation(f_inpt) and (not inpt_info.mutates_data):
                    assert not inpt_info.requires_grad, 'Found a graph input that requires_grad and was mutated in the backward. This is not supported'
                    before.copy_(after)
            tangents_before = args[1]
            tangents_after = pytree.tree_map(from_fun, f_args[1])
            for f_inpt, before, after in zip(f_args[1], tangents_before, tangents_after):
                assert not has_metadata_mutation(f_inpt, before, check_only_storage_mutation=False) and (not has_data_mutation(f_inpt)), 'Found an input to the backward that was mutated during the backward pass. This is not supported'
        if aot_config.keep_inference_input_mutations:
            for i, (inpt_old, inpt_f) in enumerate(zip(args, f_args) if not trace_joint else zip(args[0], f_args[0])):
                if not isinstance(inpt_f, torch.Tensor):
                    continue
                assert is_fun(inpt_f)
                inpt_new = from_fun(inpt_f)
                if meta.input_info[i].mutation_type == MutationType.MUTATED_IN_GRAPH:
                    if meta.input_info[i].mutations_hidden_from_autograd:
                        with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(inpt_old):
                            inpt_old.copy_(inpt_new)
                    elif meta.input_info[i].mutations_under_no_grad_or_inference_mode:
                        with torch.no_grad():
                            inpt_old.copy_(inpt_new)
                    else:
                        inpt_old.copy_(inpt_new)
        return pytree.tree_map(from_fun, f_outs)

    def joint_helper(primals, tangents):
        return _functionalized_f_helper(primals, tangents)

    def fwd_helper(*args):
        return _functionalized_f_helper(*args)
    helper = joint_helper if trace_joint else fwd_helper
    if config.functionalize_rng_ops:
        helper, args = create_functionalized_rng_ops_wrapper(helper, args, trace_joint)
    return (helper, args)