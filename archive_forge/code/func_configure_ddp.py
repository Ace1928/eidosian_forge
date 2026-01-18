import io
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.accelerators.xla import _XLA_AVAILABLE, _XLA_GREATER_EQUAL_2_1
from lightning_fabric.plugins import XLACheckpointIO
from lightning_fabric.plugins.environments import XLAEnvironment
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.types import _PATH, ReduceOp
from pytorch_lightning.plugins import XLAPrecision
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.launchers.xla import _XLALauncher
from pytorch_lightning.strategies.strategy import TBroadcast
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import find_shared_parameters, set_shared_parameters
from pytorch_lightning.utilities.rank_zero import rank_zero_only
@override
def configure_ddp(self) -> None:
    pass