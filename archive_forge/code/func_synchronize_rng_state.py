import random
from typing import List, Optional, Union
import numpy as np
import torch
from ..state import AcceleratorState
from .constants import CUDA_DISTRIBUTED_TYPES
from .dataclasses import DistributedType, RNGType
from .imports import is_npu_available, is_torch_xla_available, is_xpu_available
def synchronize_rng_state(rng_type: Optional[RNGType]=None, generator: Optional[torch.Generator]=None):
    if rng_type == RNGType.TORCH:
        rng_state = torch.get_rng_state()
    elif rng_type == RNGType.CUDA:
        rng_state = torch.cuda.get_rng_state()
    elif rng_type == RNGType.XLA:
        assert is_torch_xla_available(), "Can't synchronize XLA seeds as torch_xla is unavailable."
        rng_state = torch.tensor(xm.get_rng_state())
    elif rng_type == RNGType.NPU:
        assert is_npu_available(), "Can't synchronize NPU seeds on an environment without NPUs."
        rng_state = torch.npu.get_rng_state()
    elif rng_type == RNGType.XPU:
        assert is_xpu_available(), "Can't synchronize XPU seeds on an environment without XPUs."
        rng_state = torch.xpu.get_rng_state()
    elif rng_type == RNGType.GENERATOR:
        assert generator is not None, 'Need a generator to synchronize its seed.'
        rng_state = generator.get_state()
    state = AcceleratorState()
    if state.distributed_type == DistributedType.XLA:
        rng_state = rng_state.to(xm.xla_device())
        xm.collective_broadcast([rng_state])
        xm.mark_step()
        rng_state = rng_state.cpu()
    elif state.distributed_type in CUDA_DISTRIBUTED_TYPES or state.distributed_type == DistributedType.MULTI_NPU or state.distributed_type == DistributedType.MULTI_XPU:
        rng_state = rng_state.to(state.device)
        torch.distributed.broadcast(rng_state, 0)
        rng_state = rng_state.cpu()
    elif state.distributed_type == DistributedType.MULTI_CPU:
        torch.distributed.broadcast(rng_state, 0)
    if rng_type == RNGType.TORCH:
        torch.set_rng_state(rng_state)
    elif rng_type == RNGType.CUDA:
        torch.cuda.set_rng_state(rng_state)
    elif rng_type == RNGType.NPU:
        torch.npu.set_rng_state(rng_state)
    elif rng_type == RNGType.XPU:
        torch.xpu.set_rng_state(rng_state)
    elif rng_type == RNGType.XLA:
        xm.set_rng_state(rng_state.item())
    elif rng_type == RNGType.GENERATOR:
        generator.set_state(rng_state)