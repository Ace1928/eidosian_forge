import io
import numpy as np
import pyarrow as pa
from pyarrow.tests import util
def alltypes_sample(size=10000, seed=0, categorical=False):
    import pandas as pd
    np.random.seed(seed)
    arrays = {'uint8': np.arange(size, dtype=np.uint8), 'uint16': np.arange(size, dtype=np.uint16), 'uint32': np.arange(size, dtype=np.uint32), 'uint64': np.arange(size, dtype=np.uint64), 'int8': np.arange(size, dtype=np.int16), 'int16': np.arange(size, dtype=np.int16), 'int32': np.arange(size, dtype=np.int32), 'int64': np.arange(size, dtype=np.int64), 'float32': np.arange(size, dtype=np.float32), 'float64': np.arange(size, dtype=np.float64), 'bool': np.random.randn(size) > 0, 'datetime_ms': np.arange('2016-01-01T00:00:00.001', size, dtype='datetime64[ms]'), 'datetime_us': np.arange('2016-01-01T00:00:00.000001', size, dtype='datetime64[us]'), 'datetime_ns': np.arange('2016-01-01T00:00:00.000000001', size, dtype='datetime64[ns]'), 'timedelta': np.arange(0, size, dtype='timedelta64[s]'), 'str': pd.Series([str(x) for x in range(size)]), 'empty_str': [''] * size, 'str_with_nulls': [None] + [str(x) for x in range(size - 2)] + [None], 'null': [None] * size, 'null_list': [None] * 2 + [[None] * (x % 4) for x in range(size - 2)]}
    if categorical:
        arrays['str_category'] = arrays['str'].astype('category')
    return pd.DataFrame(arrays)