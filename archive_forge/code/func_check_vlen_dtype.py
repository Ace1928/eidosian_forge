from __future__ import annotations
from functools import partial
import numpy as np
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.utils import module_available
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def check_vlen_dtype(dtype):
    if dtype.kind != 'O' or dtype.metadata is None:
        return None
    else:
        return dtype.metadata.get('element_type', dtype.metadata.get('vlen'))