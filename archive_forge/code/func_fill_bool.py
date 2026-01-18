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
def fill_bool(x, left=None):
    if x.dtype.kind in 'cfO':
        mask = isna(x)
        if mask.any():
            x = x.astype(object)
            x[mask] = False
    if left is None or left.dtype.kind == 'b':
        x = x.astype(bool)
    return x