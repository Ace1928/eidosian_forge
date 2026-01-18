from __future__ import annotations
import operator
from typing import (
import numpy as np
from pandas._libs import lib
from pandas._libs.missing import is_matching_na
from pandas._libs.sparse import SparseIndex
import pandas._libs.testing as _testing
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
from pandas import (
from pandas.core.arrays import (
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.arrays.string_ import StringDtype
from pandas.core.indexes.api import safe_sort_index
from pandas.io.formats.printing import pprint_thing
def is_extension_array_dtype_and_needs_i8_conversion(left_dtype: DtypeObj, right_dtype: DtypeObj) -> bool:
    """
    Checks that we have the combination of an ExtensionArraydtype and
    a dtype that should be converted to int64

    Returns
    -------
    bool

    Related to issue #37609
    """
    return isinstance(left_dtype, ExtensionDtype) and needs_i8_conversion(right_dtype)