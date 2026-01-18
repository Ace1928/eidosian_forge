from typing import Any, Dict, Optional, Union
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def has_dataloader_changed(self, dataloader_idx: int) -> bool:
    old_dataloader_idx = self._current_eval_dataloader_idx
    self._current_eval_dataloader_idx = dataloader_idx
    return old_dataloader_idx != dataloader_idx