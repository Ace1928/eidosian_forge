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
def create_functionalized_rng_ops_wrapper(func, args, trace_joint=True) -> Any:
    fake_mode = detect_fake_mode()
    if fake_mode is None:
        fake_mode = nullcontext()

    def override_get_rng_state(device: Union[int, str, torch.device]='cuda'):
        out = PhiloxStateTracker.get_state_as_tensor()
        return out

    def override_set_rng_state(x, device: Union[int, str, torch.device]='cuda'):
        PhiloxStateTracker.set_state_from_tensor(x)

    def append_rng_offsets(args):
        if trace_joint:
            return ((*args[0], PhiloxStateTracker.get_updated_fwd_offset()), (*args[1], PhiloxStateTracker.get_updated_bwd_offset()))
        else:
            return (*args, PhiloxStateTracker.get_updated_fwd_offset())

    def traced_joint(primals, tangents, fwd_seed, fwd_base_offset, bwd_seed, bwd_base_offset):
        with patch('torch.cuda.get_rng_state', override_get_rng_state), patch('torch.cuda.set_rng_state', override_set_rng_state):
            return append_rng_offsets(func(primals, tangents))

    def traced_forward(*primals_fwd_seed_fwd_base_offset):
        with patch('torch.cuda.get_rng_state', override_get_rng_state), patch('torch.cuda.set_rng_state', override_set_rng_state):
            return append_rng_offsets(func(*primals_fwd_seed_fwd_base_offset[:-2]))
    if trace_joint:
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
        bwd_seed, bwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, 'forward')
        PhiloxStateTracker.record_state(bwd_seed, bwd_base_offset, 'backward')
        return (traced_joint, (*args, fwd_seed, fwd_base_offset, bwd_seed, bwd_base_offset))
    else:
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, 'forward')
        return (traced_forward, (*args, fwd_seed, fwd_base_offset))