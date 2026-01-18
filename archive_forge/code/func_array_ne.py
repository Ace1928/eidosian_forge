from __future__ import annotations
import warnings
from typing import Callable
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.core.utils import is_duck_array, module_available
from xarray.namedarray import pycompat
from xarray.core.options import OPTIONS
def array_ne(self, other):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'elementwise comparison failed')
        return _ensure_bool_is_ndarray(self != other, self, other)