import logging
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_warn
@property
@override
def state_key(self) -> str:
    return self._generate_state_key(monitor=self.monitor, mode=self.mode)