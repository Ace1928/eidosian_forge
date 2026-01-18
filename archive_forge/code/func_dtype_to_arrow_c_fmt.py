from __future__ import annotations
import typing
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.dtypes import (
def dtype_to_arrow_c_fmt(dtype: DtypeObj) -> str:
    """
    Represent pandas `dtype` as a format string in Apache Arrow C notation.

    Parameters
    ----------
    dtype : np.dtype
        Datatype of pandas DataFrame to represent.

    Returns
    -------
    str
        Format string in Apache Arrow C notation of the given `dtype`.
    """
    if isinstance(dtype, CategoricalDtype):
        return ArrowCTypes.INT64
    elif dtype == np.dtype('O'):
        return ArrowCTypes.STRING
    elif isinstance(dtype, ArrowDtype):
        import pyarrow as pa
        pa_type = dtype.pyarrow_dtype
        if pa.types.is_decimal(pa_type):
            return f'd:{pa_type.precision},{pa_type.scale}'
        elif pa.types.is_timestamp(pa_type) and pa_type.tz is not None:
            return f'ts{pa_type.unit[0]}:{pa_type.tz}'
        format_str = PYARROW_CTYPES.get(str(pa_type), None)
        if format_str is not None:
            return format_str
    format_str = getattr(ArrowCTypes, dtype.name.upper(), None)
    if format_str is not None:
        return format_str
    if lib.is_np_dtype(dtype, 'M'):
        resolution = np.datetime_data(dtype)[0][0]
        return ArrowCTypes.TIMESTAMP.format(resolution=resolution, tz='')
    elif isinstance(dtype, DatetimeTZDtype):
        return ArrowCTypes.TIMESTAMP.format(resolution=dtype.unit[0], tz=dtype.tz)
    raise NotImplementedError(f'Conversion of {dtype} to Arrow C format string is not implemented.')