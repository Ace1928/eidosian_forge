from __future__ import annotations
import string
from typing import cast
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_string_dtype
from pandas.core.arrays import ArrowStringArray
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.extension import base
def maybe_split_array(arr, chunked):
    if not chunked:
        return arr
    elif arr.dtype.storage != 'pyarrow':
        return arr
    pa = pytest.importorskip('pyarrow')
    arrow_array = arr._pa_array
    split = len(arrow_array) // 2
    arrow_array = pa.chunked_array([*arrow_array[:split].chunks, *arrow_array[split:].chunks])
    assert arrow_array.num_chunks == 2
    return type(arr)(arrow_array)