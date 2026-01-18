from __future__ import annotations
import functools
import operator
import re
import textwrap
from typing import (
import unicodedata
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas.compat import (
from pandas.util._decorators import doc
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.algorithms import map_array
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import (
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.indexers import (
from pandas.core.strings.base import BaseStringArrayMethods
from pandas.io._util import _arrow_dtype_mapping
from pandas.tseries.frequencies import to_offset
@doc(ExtensionArray.duplicated)
def duplicated(self, keep: Literal['first', 'last', False]='first') -> npt.NDArray[np.bool_]:
    pa_type = self._pa_array.type
    if pa.types.is_floating(pa_type) or pa.types.is_integer(pa_type):
        values = self.to_numpy(na_value=0)
    elif pa.types.is_boolean(pa_type):
        values = self.to_numpy(na_value=False)
    elif pa.types.is_temporal(pa_type):
        if pa_type.bit_width == 32:
            pa_type = pa.int32()
        else:
            pa_type = pa.int64()
        arr = self.astype(ArrowDtype(pa_type))
        values = arr.to_numpy(na_value=0)
    else:
        values = self.factorize()[0]
    mask = self.isna() if self._hasna else None
    return algos.duplicated(values, keep=keep, mask=mask)