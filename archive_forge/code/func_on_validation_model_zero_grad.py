from typing import Any, Dict, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
def on_validation_model_zero_grad(self) -> None:
    """Called by the training loop to release gradients before entering the validation loop."""
    zero_grad_kwargs = {} if _TORCH_GREATER_EQUAL_2_0 else {'set_to_none': True}
    self.zero_grad(**zero_grad_kwargs)