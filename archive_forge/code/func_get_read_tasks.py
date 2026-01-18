import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, List, Optional
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def get_read_tasks(self, parallelism: int) -> List[ReadTask]:

    def fallback_read_fn() -> Iterable[Block]:
        with _connect(self.connection_factory) as cursor:
            cursor.execute(self.sql)
            block = _cursor_to_block(cursor)
            return [block]
    if parallelism == 1:
        metadata = BlockMetadata(None, None, None, None, None)
        return [ReadTask(fallback_read_fn, metadata)]
    try:
        with _connect(self.connection_factory) as cursor:
            cursor.execute(f'SELECT * FROM ({self.sql}) as T LIMIT 1 OFFSET 0')
        is_limit_supported = True
    except Exception:
        is_limit_supported = False
    if not is_limit_supported:
        metadata = BlockMetadata(None, None, None, None, None)
        return [ReadTask(fallback_read_fn, metadata)]
    num_rows_total = self._get_num_rows()
    if num_rows_total == 0:
        return []
    parallelism = min(parallelism, math.ceil(num_rows_total / self.MIN_ROWS_PER_READ_TASK))
    num_rows_per_block = num_rows_total // parallelism
    num_blocks_with_extra_row = num_rows_total % parallelism
    sample_block_accessor = BlockAccessor.for_block(self._get_sample_block())
    estimated_size_bytes_per_row = math.ceil(sample_block_accessor.size_bytes() / sample_block_accessor.num_rows())
    sample_block_schema = sample_block_accessor.schema()
    tasks = []
    offset = 0
    for i in range(parallelism):
        num_rows = num_rows_per_block
        if i < num_blocks_with_extra_row:
            num_rows += 1
        read_fn = self._create_read_fn(num_rows, offset)
        metadata = BlockMetadata(num_rows, estimated_size_bytes_per_row * num_rows, sample_block_schema, None, None)
        tasks.append(ReadTask(read_fn, metadata))
        offset += num_rows
    return tasks