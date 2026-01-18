from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
@pytest.mark.pandas
def test_multiple_path_types(tempdir):
    path = tempdir / 'zzz.parquet'
    df = pd.DataFrame({'x': np.arange(10, dtype=np.int64)})
    _write_table(df, path)
    table_read = _read_table(path)
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)
    path = str(tempdir) + 'zzz.parquet'
    df = pd.DataFrame({'x': np.arange(10, dtype=np.int64)})
    _write_table(df, path)
    table_read = _read_table(path)
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)