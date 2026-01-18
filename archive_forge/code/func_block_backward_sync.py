from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
import torch
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.utilities.distributed import ReduceOp, _all_gather_ddp_if_available
from pytorch_lightning.plugins import LayerSync
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.strategies.strategy import Strategy
@contextmanager
def block_backward_sync(self) -> Generator:
    """Blocks ddp sync gradients behaviour on backwards pass.

        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off

        """
    if isinstance(self.model, pl.utilities.types.DistributedDataParallel):
        with self.model.no_sync():
            yield None
    else:
        yield None