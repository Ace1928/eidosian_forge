from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.lib import i8max
from pandas._libs.tslibs import (
def generate_regular_range(start: Timestamp | Timedelta | None, end: Timestamp | Timedelta | None, periods: int | None, freq: BaseOffset, unit: str='ns') -> npt.NDArray[np.intp]:
    """
    Generate a range of dates or timestamps with the spans between dates
    described by the given `freq` DateOffset.

    Parameters
    ----------
    start : Timedelta, Timestamp or None
        First point of produced date range.
    end : Timedelta, Timestamp or None
        Last point of produced date range.
    periods : int or None
        Number of periods in produced date range.
    freq : Tick
        Describes space between dates in produced date range.
    unit : str, default "ns"
        The resolution the output is meant to represent.

    Returns
    -------
    ndarray[np.int64]
        Representing the given resolution.
    """
    istart = start._value if start is not None else None
    iend = end._value if end is not None else None
    freq.nanos
    td = Timedelta(freq)
    b: int
    e: int
    try:
        td = td.as_unit(unit, round_ok=False)
    except ValueError as err:
        raise ValueError(f'freq={freq} is incompatible with unit={unit}. Use a lower freq or a higher unit instead.') from err
    stride = int(td._value)
    if periods is None and istart is not None and (iend is not None):
        b = istart
        e = b + (iend - b) // stride * stride + stride // 2 + 1
    elif istart is not None and periods is not None:
        b = istart
        e = _generate_range_overflow_safe(b, periods, stride, side='start')
    elif iend is not None and periods is not None:
        e = iend + stride
        b = _generate_range_overflow_safe(e, periods, stride, side='end')
    else:
        raise ValueError("at least 'start' or 'end' should be specified if a 'period' is given.")
    with np.errstate(over='raise'):
        try:
            values = np.arange(b, e, stride, dtype=np.int64)
        except FloatingPointError:
            xdr = [b]
            while xdr[-1] != e:
                xdr.append(xdr[-1] + stride)
            values = np.array(xdr[:-1], dtype=np.int64)
    return values