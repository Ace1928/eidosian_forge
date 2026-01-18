import contextlib
import warnings
from typing import Dict, List, Optional
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._tensor.placement_types import DTensorSpec, Shard
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
def rng_state_is_sync(self, name) -> bool:
    return name in self.rng_states