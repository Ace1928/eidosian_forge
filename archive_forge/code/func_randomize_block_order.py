import math
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_allocation
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats, _get_or_create_stats_actor
from ray.data._internal.util import _split_list
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.datasource import ReadTask
from ray.types import ObjectRef
def randomize_block_order(self, seed: Optional[int]=None) -> 'LazyBlockList':
    """Randomizes the order of the blocks.

        Args:
            seed: Fix the random seed to use, otherwise one will be chosen
                based on system randomness.
        """
    import random
    if seed is not None:
        random.seed(seed)
    zipped = list(zip(self._tasks, self._block_partition_refs, self._block_partition_meta_refs, self._cached_metadata))
    random.shuffle(zipped)
    tasks, block_partition_refs, block_partition_meta_refs, cached_metadata = map(list, zip(*zipped))
    return LazyBlockList(tasks, block_partition_refs=block_partition_refs, block_partition_meta_refs=block_partition_meta_refs, cached_metadata=cached_metadata, ray_remote_args=self._remote_args.copy(), owned_by_consumer=self._owned_by_consumer, stats_uuid=self._stats_uuid)