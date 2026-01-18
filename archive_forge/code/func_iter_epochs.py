import abc
import time
from typing import (
import numpy as np
from ray.data._internal.block_batching.iter_batches import iter_batches
from ray.data._internal.stats import DatasetStats, StatsManager
from ray.data.block import (
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
def iter_epochs(self, max_epoch: int=-1) -> None:
    raise DeprecationWarning('If you are using Ray Train, ray.train.get_dataset_shard() returns a ray.data.DataIterator instead of a DatasetPipeline as of Ray 2.3. To iterate over one epoch of data, use iter_batches(), iter_torch_batches(), or to_tf().')