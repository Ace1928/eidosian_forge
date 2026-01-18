from typing import Any, Dict, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
def on_validation_model_train(self) -> None:
    """Called when the validation loop ends.

        The validation loop by default restores the `training` mode of the LightningModule to what it was before
        starting validation. Override this hook to change the behavior. See also
        :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_validation_model_eval`.

        """
    self.trainer.model.train()