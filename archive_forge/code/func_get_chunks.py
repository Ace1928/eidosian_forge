from __future__ import annotations
from typing import Any
import numpy as np
from pandas._libs.lib import infer_dtype
from pandas._libs.tslibs import iNaT
from pandas.errors import NoBufferPresent
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.core.interchange.buffer import PandasBuffer
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.utils import (
def get_chunks(self, n_chunks: int | None=None):
    """
        Return an iterator yielding the chunks.
        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
    if n_chunks and n_chunks > 1:
        size = len(self._col)
        step = size // n_chunks
        if size % n_chunks != 0:
            step += 1
        for start in range(0, step * n_chunks, step):
            yield PandasColumn(self._col.iloc[start:start + step], self._allow_copy)
    else:
        yield self