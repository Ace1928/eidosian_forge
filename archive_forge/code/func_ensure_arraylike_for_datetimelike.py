from __future__ import annotations
from datetime import (
from functools import wraps
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas._libs.tslibs.timedeltas import get_unit_for_round
from pandas._libs.tslibs.timestamps import integer_op_not_supported
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.algorithms import (
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import (
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.ops.invalid import (
from pandas.tseries import frequencies
def ensure_arraylike_for_datetimelike(data, copy: bool, cls_name: str) -> tuple[ArrayLike, bool]:
    if not hasattr(data, 'dtype'):
        if not isinstance(data, (list, tuple)) and np.ndim(data) == 0:
            data = list(data)
        data = construct_1d_object_array_from_listlike(data)
        copy = False
    elif isinstance(data, ABCMultiIndex):
        raise TypeError(f'Cannot create a {cls_name} from a MultiIndex.')
    else:
        data = extract_array(data, extract_numpy=True)
    if isinstance(data, IntegerArray) or (isinstance(data, ArrowExtensionArray) and data.dtype.kind in 'iu'):
        data = data.to_numpy('int64', na_value=iNaT)
        copy = False
    elif isinstance(data, ArrowExtensionArray):
        data = data._maybe_convert_datelike_array()
        data = data.to_numpy()
        copy = False
    elif not isinstance(data, (np.ndarray, ExtensionArray)):
        data = np.asarray(data)
    elif isinstance(data, ABCCategorical):
        data = data.categories.take(data.codes, fill_value=NaT)._values
        copy = False
    return (data, copy)