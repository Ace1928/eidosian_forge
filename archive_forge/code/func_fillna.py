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
@doc(ExtensionArray.fillna)
def fillna(self, value: object | ArrayLike | None=None, method: FillnaOptions | None=None, limit: int | None=None, copy: bool=True) -> Self:
    value, method = validate_fillna_kwargs(value, method)
    if not self._hasna:
        return self.copy()
    if limit is not None:
        return super().fillna(value=value, method=method, limit=limit, copy=copy)
    if method is not None:
        return super().fillna(method=method, limit=limit, copy=copy)
    if isinstance(value, (np.ndarray, ExtensionArray)):
        if len(value) != len(self):
            raise ValueError(f"Length of 'value' does not match. Got ({len(value)})  expected {len(self)}")
    try:
        fill_value = self._box_pa(value, pa_type=self._pa_array.type)
    except pa.ArrowTypeError as err:
        msg = f"Invalid value '{str(value)}' for dtype {self.dtype}"
        raise TypeError(msg) from err
    try:
        return type(self)(pc.fill_null(self._pa_array, fill_value=fill_value))
    except pa.ArrowNotImplementedError:
        pass
    return super().fillna(value=value, method=method, limit=limit, copy=copy)