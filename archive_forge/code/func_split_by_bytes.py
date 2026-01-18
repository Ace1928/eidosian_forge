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
def split_by_bytes(self, bytes_per_split: int) -> List['BlockList']:
    output = []
    cur_tasks, cur_blocks, cur_blocks_meta, cur_cached_meta = ([], [], [], [])
    cur_size = 0
    for t, b, bm, c in zip(self._tasks, self._block_partition_refs, self._block_partition_meta_refs, self._cached_metadata):
        m = t.get_metadata()
        if m.size_bytes is None:
            raise RuntimeError('Block has unknown size, cannot use split_by_bytes()')
        size = m.size_bytes
        if cur_blocks and cur_size + size > bytes_per_split:
            output.append(LazyBlockList(cur_tasks, cur_blocks, cur_blocks_meta, cur_cached_meta, owned_by_consumer=self._owned_by_consumer))
            cur_tasks, cur_blocks, cur_blocks_meta, cur_cached_meta = ([], [], [], [])
            cur_size = 0
        cur_tasks.append(t)
        cur_blocks.append(b)
        cur_blocks_meta.append(bm)
        cur_cached_meta.append(c)
        cur_size += size
    if cur_blocks:
        output.append(LazyBlockList(cur_tasks, cur_blocks, cur_blocks_meta, cur_cached_meta, owned_by_consumer=self._owned_by_consumer))
    return output