from io import StringIO
import re
from string import ascii_uppercase as uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_info_memory_usage():
    dtypes = ['int64', 'float64', 'datetime64[ns]', 'timedelta64[ns]', 'complex128', 'object', 'bool']
    data = {}
    n = 10
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.randint(2, size=n).astype(dtype)
    df = DataFrame(data)
    buf = StringIO()
    df.info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()
    assert 'memory usage: ' in res[-1]
    df.info(buf=buf, memory_usage=False)
    res = buf.getvalue().splitlines()
    assert 'memory usage: ' not in res[-1]
    df.info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()
    assert re.match('memory usage: [^+]+\\+', res[-1])
    df.iloc[:, :5].info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()
    assert not re.match('memory usage: [^+]+\\+', res[-1])
    dtypes = ['int64', 'int64', 'int64', 'float64']
    data = {}
    n = 100
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.randint(2, size=n).astype(dtype)
    df = DataFrame(data)
    df.columns = dtypes
    df_with_object_index = DataFrame({'a': [1]}, index=['foo'])
    df_with_object_index.info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()
    assert re.match('memory usage: [^+]+\\+', res[-1])
    df_with_object_index.info(buf=buf, memory_usage='deep')
    res = buf.getvalue().splitlines()
    assert re.match('memory usage: [^+]+$', res[-1])
    df_size = df.memory_usage().sum()
    exp_size = len(dtypes) * n * 8 + df.index.nbytes
    assert df_size == exp_size
    size_df = np.size(df.columns.values) + 1
    assert size_df == np.size(df.memory_usage())
    assert df.memory_usage().sum() == df.memory_usage(deep=True).sum()
    DataFrame(1, index=['a'], columns=['A']).memory_usage(index=True)
    DataFrame(1, index=['a'], columns=['A']).index.nbytes
    df = DataFrame(data=1, index=MultiIndex.from_product([['a'], range(1000)]), columns=['A'])
    df.index.nbytes
    df.memory_usage(index=True)
    df.index.values.nbytes
    mem = df.memory_usage(deep=True).sum()
    assert mem > 0