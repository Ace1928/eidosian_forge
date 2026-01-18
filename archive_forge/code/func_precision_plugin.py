import os
from typing import Optional, Union
import torch
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.accelerators.xla import _XLA_AVAILABLE
from lightning_fabric.plugins import XLACheckpointIO
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.plugins.precision.xla import XLAPrecision
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import find_shared_parameters, set_shared_parameters
@precision_plugin.setter
@override
def precision_plugin(self, precision_plugin: Optional[XLAPrecision]) -> None:
    if precision_plugin is not None and (not isinstance(precision_plugin, XLAPrecision)):
        raise TypeError(f'The XLA strategy can only work with the `XLAPrecision` plugin, found {precision_plugin}')
    self._precision_plugin = precision_plugin