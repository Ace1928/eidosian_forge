from typing import Any, Dict, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
def on_predict_model_eval(self) -> None:
    """Called when the predict loop starts.

        The predict loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
        to change the behavior.

        """
    self.trainer.model.eval()