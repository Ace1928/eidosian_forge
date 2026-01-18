from typing import Any, Dict
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _LIGHTNING_COLOSSALAI_AVAILABLE
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def get_accumulate_grad_batches(self, epoch: int) -> int:
    accumulate_grad_batches = 1
    for iter_epoch in reversed(self.epochs):
        if epoch >= iter_epoch:
            accumulate_grad_batches = self.scheduling[iter_epoch]
            break
    return accumulate_grad_batches