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
def na_logical_op(x: np.ndarray, y, op):
    try:
        result = op(x, y)
    except TypeError:
        if isinstance(y, np.ndarray):
            assert not (x.dtype.kind == 'b' and y.dtype.kind == 'b')
            x = ensure_object(x)
            y = ensure_object(y)
            result = libops.vec_binop(x.ravel(), y.ravel(), op)
        else:
            assert lib.is_scalar(y)
            if not isna(y):
                y = bool(y)
            try:
                result = libops.scalar_binop(x, y, op)
            except (TypeError, ValueError, AttributeError, OverflowError, NotImplementedError) as err:
                typ = type(y).__name__
                raise TypeError(f"Cannot perform '{op.__name__}' with a dtyped [{x.dtype}] array and scalar of type [{typ}]") from err
    return result.reshape(x.shape)