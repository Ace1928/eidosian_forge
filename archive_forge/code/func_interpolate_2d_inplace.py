from __future__ import annotations
from functools import wraps
from typing import (
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
def interpolate_2d_inplace(data: np.ndarray, index: Index, axis: AxisInt, method: str='linear', limit: int | None=None, limit_direction: str='forward', limit_area: str | None=None, fill_value: Any | None=None, mask=None, **kwargs) -> None:
    """
    Column-wise application of _interpolate_1d.

    Notes
    -----
    Alters 'data' in-place.

    The signature does differ from _interpolate_1d because it only
    includes what is needed for Block.interpolate.
    """
    clean_interp_method(method, index, **kwargs)
    if is_valid_na_for_dtype(fill_value, data.dtype):
        fill_value = na_value_for_dtype(data.dtype, compat=False)
    if method == 'time':
        if not needs_i8_conversion(index.dtype):
            raise ValueError('time-weighted interpolation only works on Series or DataFrames with a DatetimeIndex')
        method = 'values'
    limit_direction = validate_limit_direction(limit_direction)
    limit_area_validated = validate_limit_area(limit_area)
    limit = algos.validate_limit(nobs=None, limit=limit)
    indices = _index_to_interp_indices(index, method)

    def func(yvalues: np.ndarray) -> None:
        _interpolate_1d(indices=indices, yvalues=yvalues, method=method, limit=limit, limit_direction=limit_direction, limit_area=limit_area_validated, fill_value=fill_value, bounds_error=False, mask=mask, **kwargs)
    np.apply_along_axis(func, axis, data)