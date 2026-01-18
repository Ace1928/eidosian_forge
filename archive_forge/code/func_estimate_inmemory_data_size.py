import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, List, Optional
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def estimate_inmemory_data_size(self) -> Optional[int]:
    None