import builtins
from copy import copy
from typing import Iterable, List, Optional, Tuple
import numpy as np
from ray.data._internal.util import _check_pyarrow_version
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def make_blocks(start: int, count: int, target_rows_per_block: int) -> Iterable[Block]:
    while count > 0:
        num_rows = min(count, target_rows_per_block)
        yield make_block(start, num_rows)
        start += num_rows
        count -= num_rows