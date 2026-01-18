from typing import Any, Dict
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _LIGHTNING_COLOSSALAI_AVAILABLE
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def going_to_accumulate_grad_batches(self) -> bool:
    return any((v > 1 for v in self.scheduling.values()))