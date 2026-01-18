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
def iter_blocks_with_metadata(self, block_for_metadata: bool=False) -> Iterator[Tuple[ObjectRef[Block], BlockMetadata]]:
    """Iterate over the blocks along with their metadata.

        Note that, if block_for_metadata is False (default), this iterator returns
        pre-read metadata from the ReadTasks given to this LazyBlockList so it doesn't
        have to block on the execution of the read tasks. Therefore, the metadata may be
        under-specified, e.g. missing schema or the number of rows. If fully-specified
        block metadata is required, pass block_for_metadata=True. When dynamic block
        splitting is enabled, always block on the execution of the read tasks.

        The length of this iterator is not known until execution.

        Args:
            block_for_metadata: Whether we should block on the execution of read tasks
                in order to obtain fully-specified block metadata.

        Returns:
            An iterator of block references and the corresponding block metadata.
        """
    outer = self

    class Iter:

        def __init__(self):
            self._base_iter = outer._iter_block_partition_refs()
            self._pos = -1
            self._buffer = []

        def __iter__(self):
            return self

        def __next__(self):
            while not self._buffer:
                self._pos += 1
                generator_ref, _ = next(self._base_iter)
                generator = ray.get(generator_ref)
                refs = list(generator)
                metadata = ray.get(refs.pop(-1))
                assert len(metadata) == len(refs)
                for block_ref, meta in zip(refs, metadata):
                    self._buffer.append((block_ref, meta))
            return self._buffer.pop(0)
    return Iter()