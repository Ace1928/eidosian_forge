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
def fn_prepped_for_autograd(fn: Callable, meta: ViewAndMutationMeta) -> Any:

    def inner_fn(*args):
        args_maybe_cloned = [maybe_to_fresh_input(i, t, meta) for i, t in enumerate(args)]
        outs = fn(*args_maybe_cloned)
        assert isinstance(outs, (tuple, list))
        outs = list(outs)
        assert len(meta.output_info) == len(outs)
        mutated_inputs_to_return = [x for i, x in enumerate(args_maybe_cloned) if i in meta.mutated_inp_runtime_indices]
        intermediate_bases = []
        for i, (o, info) in enumerate(zip(outs, meta.output_info)):
            if info.output_type == OutputType.alias_of_intermediate_save_as_output:
                intermediate_bases.append(o._base)
        assert meta.num_intermediate_bases == len(intermediate_bases)
        fw_outs_to_return = (*mutated_inputs_to_return, *outs, *intermediate_bases)
        mutated_inputs_grad_mask = [meta.input_info[meta.mutated_inp_runtime_indices[i]].mutates_data and meta.input_info[meta.mutated_inp_runtime_indices[i]].requires_grad for i, x in enumerate(mutated_inputs_to_return)]
        output_grad_mask = [meta.output_info[i].output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view] and issubclass(meta.output_info[i].raw_type, Tensor) and meta.output_info[i].requires_grad for i, x in enumerate(outs)]
        intermediate_base_grad_mask = [True for _ in range(len(intermediate_bases))]
        out_grad_mask = mutated_inputs_grad_mask + output_grad_mask + intermediate_base_grad_mask
        assert len(out_grad_mask) == len(fw_outs_to_return)
        for arg in args_maybe_cloned:
            if not isinstance(arg, Tensor):
                continue
            sync_functional_tensor(arg)
        return (fw_outs_to_return, out_grad_mask)
    return inner_fn