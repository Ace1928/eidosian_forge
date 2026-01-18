import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def test_arrowparquet_options(fsspectest):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    pytest.importorskip('pyarrow')
    df = DataFrame({'a': [0]})
    df.to_parquet('testmem://test/test.csv', engine='pyarrow', compression=None, storage_options={'test': 'parquet_write'})
    assert fsspectest.test[0] == 'parquet_write'
    read_parquet('testmem://test/test.csv', engine='pyarrow', storage_options={'test': 'parquet_read'})
    assert fsspectest.test[0] == 'parquet_read'