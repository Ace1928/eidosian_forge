import inspect
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch.nn.utils.prune as pytorch_prune
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor, nn
from typing_extensions import TypedDict, override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_only
def apply_pruning(self, amount: Union[int, float]) -> None:
    """Applies pruning to ``parameters_to_prune``."""
    if self._verbose:
        prev_stats = [self._get_pruned_stats(m, n) for m, n in self._parameters_to_prune]
    if self._use_global_unstructured:
        self._apply_global_pruning(amount)
    else:
        self._apply_local_pruning(amount)
    if self._verbose:
        curr_stats = [self._get_pruned_stats(m, n) for m, n in self._parameters_to_prune]
        self._log_sparsity_stats(prev_stats, curr_stats, amount=amount)