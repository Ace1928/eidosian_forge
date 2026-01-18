from __future__ import annotations
from collections import abc
from datetime import datetime
import functools
from itertools import zip_longest
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import BlockValuesRefs
import pandas._libs.join as libjoin
from pandas._libs.lib import (
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.missing import clean_reindex_fill_method
from pandas.core.ops import get_op_result_name
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.core.strings.accessor import StringMethods
from pandas.io.formats.printing import (
def get_values_for_csv(values: ArrayLike, *, date_format, na_rep: str='nan', quoting=None, float_format=None, decimal: str='.') -> npt.NDArray[np.object_]:
    """
    Convert to types which can be consumed by the standard library's
    csv.writer.writerows.
    """
    if isinstance(values, Categorical) and values.categories.dtype.kind in 'Mm':
        values = algos.take_nd(values.categories._values, ensure_platform_int(values._codes), fill_value=na_rep)
    values = ensure_wrapped_if_datetimelike(values)
    if isinstance(values, (DatetimeArray, TimedeltaArray)):
        if values.ndim == 1:
            result = values._format_native_types(na_rep=na_rep, date_format=date_format)
            result = result.astype(object, copy=False)
            return result
        results_converted = []
        for i in range(len(values)):
            result = values[i, :]._format_native_types(na_rep=na_rep, date_format=date_format)
            results_converted.append(result.astype(object, copy=False))
        return np.vstack(results_converted)
    elif isinstance(values.dtype, PeriodDtype):
        values = cast('PeriodArray', values)
        res = values._format_native_types(na_rep=na_rep, date_format=date_format)
        return res
    elif isinstance(values.dtype, IntervalDtype):
        values = cast('IntervalArray', values)
        mask = values.isna()
        if not quoting:
            result = np.asarray(values).astype(str)
        else:
            result = np.array(values, dtype=object, copy=True)
        result[mask] = na_rep
        return result
    elif values.dtype.kind == 'f' and (not isinstance(values.dtype, SparseDtype)):
        if float_format is None and decimal == '.':
            mask = isna(values)
            if not quoting:
                values = values.astype(str)
            else:
                values = np.array(values, dtype='object')
            values[mask] = na_rep
            values = values.astype(object, copy=False)
            return values
        from pandas.io.formats.format import FloatArrayFormatter
        formatter = FloatArrayFormatter(values, na_rep=na_rep, float_format=float_format, decimal=decimal, quoting=quoting, fixed_width=False)
        res = formatter.get_result_as_array()
        res = res.astype(object, copy=False)
        return res
    elif isinstance(values, ExtensionArray):
        mask = isna(values)
        new_values = np.asarray(values.astype(object))
        new_values[mask] = na_rep
        return new_values
    else:
        mask = isna(values)
        itemsize = writers.word_len(na_rep)
        if values.dtype != _dtype_obj and (not quoting) and itemsize:
            values = values.astype(str)
            if values.dtype.itemsize / np.dtype('U1').itemsize < itemsize:
                values = values.astype(f'<U{itemsize}')
        else:
            values = np.array(values, dtype='object')
        values[mask] = na_rep
        values = values.astype(object, copy=False)
        return values