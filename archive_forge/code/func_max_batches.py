import os
import shutil
import sys
from collections import ChainMap, OrderedDict, defaultdict
from typing import Any, DefaultDict, Iterable, Iterator, List, Optional, Tuple, Union
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.utilities.data import _set_sampler_epoch
from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE
from pytorch_lightning.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from pytorch_lightning.loops.loop import _Loop
from pytorch_lightning.loops.progress import _BatchProgress
from pytorch_lightning.loops.utilities import _no_grad_context, _select_data_fetcher, _verify_dataloader_idx_requirement
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.data_connector import (
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT, _ResultCollection
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.data import has_len_all_ranks
from pytorch_lightning.utilities.exceptions import SIGTERMException
from pytorch_lightning.utilities.model_helpers import _ModuleMode, is_overridden
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
@property
def max_batches(self) -> List[Union[int, float]]:
    """The max number of batches to run per dataloader."""
    max_batches = self._max_batches
    if not self.trainer.sanity_checking:
        return max_batches
    return [min(self.trainer.num_sanity_val_steps, batches) for batches in max_batches]