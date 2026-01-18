from typing import Any, Dict, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
def configure_model(self) -> None:
    """Hook to create modules in a strategy and precision aware context.

        This is particularly useful for when using sharded strategies (FSDP and DeepSpeed), where we'd like to shard
        the model instantly to save memory and initialization time.
        For non-sharded strategies, you can choose to override this hook or to initialize your model under the
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.init_module` context manager.

        This hook is called during each of fit/val/test/predict stages in the same process, so ensure that
        implementation of this hook is **idempotent**, i.e., after the first time the hook is called, subsequent calls
        to it should be a no-op.

        """