from collections import OrderedDict
from datetime import date, time
import numpy as np
import pandas as pd
import pyarrow as pa
def dataframe_with_arrays(include_index=False):
    """
    Dataframe with numpy arrays columns of every possible primitive type.

    Returns
    -------
    df: pandas.DataFrame
    schema: pyarrow.Schema
        Arrow schema definition that is in line with the constructed df.
    """
    dtypes = [('i1', pa.int8()), ('i2', pa.int16()), ('i4', pa.int32()), ('i8', pa.int64()), ('u1', pa.uint8()), ('u2', pa.uint16()), ('u4', pa.uint32()), ('u8', pa.uint64()), ('f4', pa.float32()), ('f8', pa.float64())]
    arrays = OrderedDict()
    fields = []
    for dtype, arrow_dtype in dtypes:
        fields.append(pa.field(dtype, pa.list_(arrow_dtype)))
        arrays[dtype] = [np.arange(10, dtype=dtype), np.arange(5, dtype=dtype), None, np.arange(1, dtype=dtype)]
    fields.append(pa.field('str', pa.list_(pa.string())))
    arrays['str'] = [np.array(['1', 'ä'], dtype='object'), None, np.array(['1'], dtype='object'), np.array(['1', '2', '3'], dtype='object')]
    fields.append(pa.field('datetime64', pa.list_(pa.timestamp('ms'))))
    arrays['datetime64'] = [np.array(['2007-07-13T01:23:34.123456789', None, '2010-08-13T05:46:57.437699912'], dtype='datetime64[ms]'), None, None, np.array(['2007-07-13T02', None, '2010-08-13T05:46:57.437699912'], dtype='datetime64[ms]')]
    if include_index:
        fields.append(pa.field('__index_level_0__', pa.int64()))
    df = pd.DataFrame(arrays)
    schema = pa.schema(fields)
    return (df, schema)