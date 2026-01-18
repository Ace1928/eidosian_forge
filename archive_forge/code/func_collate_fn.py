import abc
import time
from typing import (
import numpy as np
from ray.data._internal.block_batching.iter_batches import iter_batches
from ray.data._internal.stats import DatasetStats, StatsManager
from ray.data.block import (
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
def collate_fn(batch: Union[np.ndarray, Dict[str, np.ndarray]]):
    return convert_ndarray_batch_to_torch_tensor_batch(batch, dtypes=dtypes, device=None)