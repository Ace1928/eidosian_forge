from typing import Any, Dict, Type
from torch import Tensor
from torch.optim import Optimizer
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
def on_before_optimizer_step(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', optimizer: Optimizer) -> None:
    """Called before ``optimizer.step()``."""