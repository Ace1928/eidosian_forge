from typing import Optional
import torch
from typing_extensions import override
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.accelerators.xla import _XLA_AVAILABLE
from lightning_fabric.plugins import XLAPrecision
from lightning_fabric.plugins.io.xla import XLACheckpointIO
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.strategies.single_device import SingleDeviceStrategy
from lightning_fabric.utilities.types import _DEVICE
@checkpoint_io.setter
@override
def checkpoint_io(self, io: Optional[XLACheckpointIO]) -> None:
    if io is not None and (not isinstance(io, XLACheckpointIO)):
        raise TypeError(f'The XLA strategy can only work with the `XLACheckpointIO` plugin, found {io}')
    self._checkpoint_io = io