import os
import random
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _test_dataframe
from pyarrow.tests.parquet.test_dataset import (
from pyarrow.util import guid
@pytest.mark.hdfs
@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.fastparquet
def test_fastparquet_read_with_hdfs():
    check_libhdfs_present()
    try:
        import snappy
    except ImportError:
        pytest.skip('fastparquet test requires snappy')
    import pyarrow.parquet as pq
    fastparquet = pytest.importorskip('fastparquet')
    fs = hdfs_test_client()
    df = util.make_dataframe()
    table = pa.Table.from_pandas(df)
    path = '/tmp/testing.parquet'
    with fs.open(path, 'wb') as f:
        pq.write_table(table, f)
    parquet_file = fastparquet.ParquetFile(path, open_with=fs.open)
    result = parquet_file.to_pandas()
    assert_frame_equal(result, df)