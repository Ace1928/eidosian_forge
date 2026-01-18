from abc import ABC
from typing import Any, Dict, List, Optional
import torch
from torch import Tensor
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.strategy import Strategy
from lightning_fabric.utilities.distributed import _all_gather_ddp_if_available
from lightning_fabric.utilities.types import ReduceOp
@parallel_devices.setter
def parallel_devices(self, parallel_devices: Optional[List[torch.device]]) -> None:
    self._parallel_devices = parallel_devices