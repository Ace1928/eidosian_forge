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
def ensure_metadata_for_first_block(self) -> Optional[BlockMetadata]:
    """Ensure that the metadata is fetched and set for the first block.

        This will only block execution in order to fetch the post-read metadata for the
        first block if the pre-read metadata for the first block has no schema.

        Returns:
            None if the block list is empty, the metadata for the first block otherwise.
        """
    if not self._tasks:
        return None
    metadata = self._tasks[0].get_metadata()
    if metadata.schema is not None:
        return metadata
    try:
        block_partition_ref, metadata_ref = next(self._iter_block_partition_refs())
    except (StopIteration, ValueError):
        pass
    else:
        generator = ray.get(block_partition_ref)
        blocks_ref = list(generator)
        metadata = ray.get(blocks_ref[-1])
        self._cached_metadata[0] = metadata
    return metadata