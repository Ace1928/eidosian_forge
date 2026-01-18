from __future__ import annotations
from functools import partial
import numpy as np
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.utils import module_available
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def create_vlen_dtype(element_type):
    if element_type not in (str, bytes):
        raise TypeError(f'unsupported type for vlen_dtype: {element_type!r}')
    return np.dtype('O', metadata={'element_type': element_type})