from typing import Any, Dict, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
def configure_sharded_model(self) -> None:
    """Deprecated.

        Use :meth:`~pytorch_lightning.core.hooks.ModelHooks.configure_model` instead.

        """