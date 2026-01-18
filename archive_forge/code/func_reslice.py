import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
def reslice(self, batches: Iterable[T]) -> Iterable[Iterable[T]]:
    """Reslice the batch stream into a stream of iterable of batches of the
        same keys

        :param batches: the batch stream

        :yield: an iterable of iterable of batches containing same keys
        """

    def slicer(n: int, current: Tuple[bool, T], last: Optional[Tuple[bool, T]]) -> bool:
        return current[0]

    def get_slices() -> Iterable[Tuple[bool, T]]:
        for batch in batches:
            if self.get_batch_length(batch) > 0:
                yield from self._reslice_single(batch)

    def transform(data: Iterable[Tuple[bool, T]]) -> Iterable[T]:
        for _, batch in data:
            yield batch
    for res in slice_iterable(get_slices(), slicer):
        yield transform(res)