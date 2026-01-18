import contextlib
from functools import partial
from typing import Any, Callable, Generator, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.plugins import Precision as FabricPrecision
from lightning_fabric.utilities.types import Steppable
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities import GradClipAlgorithmType
def clip_grad_by_value(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
    """Clip gradients by value."""
    parameters = self.main_params(optimizer)
    torch.nn.utils.clip_grad_value_(parameters, clip_value=clip_val)