from __future__ import annotations
import datetime
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import roperator
from pandas.core.computation import expressions
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.ops import missing
from pandas.core.ops.dispatch import should_extension_dispatch
from pandas.core.ops.invalid import invalid_comparison
def comp_method_OBJECT_ARRAY(op, x, y):
    if isinstance(y, list):
        y = construct_1d_object_array_from_listlike(y)
    if isinstance(y, (np.ndarray, ABCSeries, ABCIndex)):
        if not is_object_dtype(y.dtype):
            y = y.astype(np.object_)
        if isinstance(y, (ABCSeries, ABCIndex)):
            y = y._values
        if x.shape != y.shape:
            raise ValueError('Shapes must match', x.shape, y.shape)
        result = libops.vec_compare(x.ravel(), y.ravel(), op)
    else:
        result = libops.scalar_compare(x.ravel(), y, op)
    return result.reshape(x.shape)