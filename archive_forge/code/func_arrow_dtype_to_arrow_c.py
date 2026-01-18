import functools
import numpy as np
import pyarrow as pa
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
def arrow_dtype_to_arrow_c(dtype: pa.DataType) -> str:
    """
    Represent PyArrow `dtype` as a format string in Apache Arrow C notation.

    Parameters
    ----------
    dtype : pa.DataType
        Datatype of PyArrow table to represent.

    Returns
    -------
    str
        Format string in Apache Arrow C notation of the given `dtype`.
    """
    if pa.types.is_timestamp(dtype):
        return ArrowCTypes.TIMESTAMP.format(resolution=dtype.unit[:1], tz=dtype.tz or '')
    elif pa.types.is_date(dtype):
        return getattr(ArrowCTypes, f'DATE{dtype.bit_width}', 'DATE64')
    elif pa.types.is_time(dtype):
        return ArrowCTypes.TIME.format(resolution=getattr(dtype, 'unit', 's')[:1])
    elif pa.types.is_dictionary(dtype):
        return arrow_dtype_to_arrow_c(dtype.index_type)
    else:
        return pandas_dtype_to_arrow_c(np.dtype(dtype.to_pandas_dtype()))