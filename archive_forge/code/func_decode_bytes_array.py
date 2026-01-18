from __future__ import annotations
from functools import partial
import numpy as np
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.utils import module_available
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def decode_bytes_array(bytes_array, encoding='utf-8'):
    bytes_array = np.asarray(bytes_array)
    decoded = [x.decode(encoding) for x in bytes_array.ravel()]
    return np.array(decoded, dtype=object).reshape(bytes_array.shape)