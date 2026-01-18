import inspect
import warnings
from typing import Any, List, Optional, Set
import torch
from torch.utils.data.datapipes.iter.sharding import (
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps
def apply_shuffle_seed(datapipe: DataPipe, rng: Any) -> DataPipe:
    warnings.warn('`apply_shuffle_seed` is deprecated since 1.12 and will be removed in the future releases.\nPlease use `apply_random_seed` instead.')
    return apply_random_seed(datapipe, rng)