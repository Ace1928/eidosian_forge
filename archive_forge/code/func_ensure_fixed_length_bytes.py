from __future__ import annotations
from functools import partial
import numpy as np
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.utils import module_available
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def ensure_fixed_length_bytes(var: Variable) -> Variable:
    """Ensure that a variable with vlen bytes is converted to fixed width."""
    if check_vlen_dtype(var.dtype) == bytes:
        dims, data, attrs, encoding = unpack_for_encoding(var)
        data = np.asarray(data, dtype=np.bytes_)
        return Variable(dims, data, attrs, encoding)
    else:
        return var