import enum
import re
from typing import Optional, Union
import numpy as np
import pandas
from pandas.api.types import is_datetime64_dtype
def pandas_dtype_to_arrow_c(dtype: Union[np.dtype, pandas.CategoricalDtype]) -> str:
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
    if isinstance(dtype, pandas.CategoricalDtype):
        return ArrowCTypes.INT64
    elif dtype == pandas.api.types.pandas_dtype('O'):
        return ArrowCTypes.STRING
    format_str = getattr(ArrowCTypes, dtype.name.upper(), None)
    if format_str is not None:
        return format_str
    if is_datetime64_dtype(dtype):
        resolution = re.findall('\\[(.*)\\]', dtype.str)[0][:1]
        return ArrowCTypes.TIMESTAMP.format(resolution=resolution, tz='')
    raise NotImplementedError(f'Convertion of {dtype} to Arrow C format string is not implemented.')