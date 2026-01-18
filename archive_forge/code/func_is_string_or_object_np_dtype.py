from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import conversion
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCIndex
from pandas.core.dtypes.inference import (
def is_string_or_object_np_dtype(dtype: np.dtype) -> bool:
    """
    Faster alternative to is_string_dtype, assumes we have a np.dtype object.
    """
    return dtype == object or dtype.kind in 'SU'