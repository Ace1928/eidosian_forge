from __future__ import annotations
from datetime import timedelta
import operator
from typing import (
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.timedeltas import (
from pandas.compat.numpy import function as nv
from pandas.util._validators import validate_endpoints
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.core.ops.common import unpack_zerodim_and_defer
import textwrap
def sequence_to_td64ns(data, copy: bool=False, unit=None, errors: DateTimeErrorChoices='raise') -> tuple[np.ndarray, Tick | None]:
    """
    Parameters
    ----------
    data : list-like
    copy : bool, default False
    unit : str, optional
        The timedelta unit to treat integers as multiples of. For numeric
        data this defaults to ``'ns'``.
        Must be un-specified if the data contains a str and ``errors=="raise"``.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    converted : numpy.ndarray
        The sequence converted to a numpy array with dtype ``timedelta64[ns]``.
    inferred_freq : Tick or None
        The inferred frequency of the sequence.

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting ``errors=ignore`` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    """
    assert unit not in ['Y', 'y', 'M']
    inferred_freq = None
    if unit is not None:
        unit = parse_timedelta_unit(unit)
    data, copy = dtl.ensure_arraylike_for_datetimelike(data, copy, cls_name='TimedeltaArray')
    if isinstance(data, TimedeltaArray):
        inferred_freq = data.freq
    if data.dtype == object or is_string_dtype(data.dtype):
        data = _objects_to_td64ns(data, unit=unit, errors=errors)
        copy = False
    elif is_integer_dtype(data.dtype):
        data, copy_made = _ints_to_td64ns(data, unit=unit)
        copy = copy and (not copy_made)
    elif is_float_dtype(data.dtype):
        if isinstance(data.dtype, ExtensionDtype):
            mask = data._mask
            data = data._data
        else:
            mask = np.isnan(data)
        data = cast_from_unit_vectorized(data, unit or 'ns')
        data[mask] = iNaT
        data = data.view('m8[ns]')
        copy = False
    elif lib.is_np_dtype(data.dtype, 'm'):
        if not is_supported_dtype(data.dtype):
            new_dtype = get_supported_dtype(data.dtype)
            data = astype_overflowsafe(data, dtype=new_dtype, copy=False)
            copy = False
    else:
        raise TypeError(f'dtype {data.dtype} cannot be converted to timedelta64[ns]')
    data = np.array(data, copy=copy)
    assert data.dtype.kind == 'm'
    assert data.dtype != 'm8'
    return (data, inferred_freq)