from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
from typing import (
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.util import capitalize_first_letter
@classmethod
def from_numpy_dtype(cls, dtype: np.dtype) -> BaseMaskedDtype:
    """
        Construct the MaskedDtype corresponding to the given numpy dtype.
        """
    if dtype.kind == 'b':
        from pandas.core.arrays.boolean import BooleanDtype
        return BooleanDtype()
    elif dtype.kind in 'iu':
        from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
        return NUMPY_INT_TO_DTYPE[dtype]
    elif dtype.kind == 'f':
        from pandas.core.arrays.floating import NUMPY_FLOAT_TO_DTYPE
        return NUMPY_FLOAT_TO_DTYPE[dtype]
    else:
        raise NotImplementedError(dtype)