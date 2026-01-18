from __future__ import annotations
from decimal import Decimal
from functools import partial
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
def array_equals(left: ArrayLike, right: ArrayLike) -> bool:
    """
    ExtensionArray-compatible implementation of array_equivalent.
    """
    if left.dtype != right.dtype:
        return False
    elif isinstance(left, ABCExtensionArray):
        return left.equals(right)
    else:
        return array_equivalent(left, right, dtype_equal=True)