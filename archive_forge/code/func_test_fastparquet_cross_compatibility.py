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
@pytest.mark.fastparquet
@pytest.mark.filterwarnings('ignore:RangeIndex:FutureWarning')
@pytest.mark.filterwarnings('ignore:tostring:DeprecationWarning:fastparquet')
def test_fastparquet_cross_compatibility(tempdir):
    fp = pytest.importorskip('fastparquet')
    df = pd.DataFrame({'a': list('abc'), 'b': list(range(1, 4)), 'c': np.arange(4.0, 7.0, dtype='float64'), 'd': [True, False, True], 'e': pd.date_range('20130101', periods=3), 'f': pd.Categorical(['a', 'b', 'a'])})
    table = pa.table(df)
    file_arrow = str(tempdir / 'cross_compat_arrow.parquet')
    pq.write_table(table, file_arrow, compression=None)
    fp_file = fp.ParquetFile(file_arrow)
    df_fp = fp_file.to_pandas()
    tm.assert_frame_equal(df, df_fp)
    file_fastparquet = str(tempdir / 'cross_compat_fastparquet.parquet')
    fp.write(file_fastparquet, df)
    table_fp = pq.read_pandas(file_fastparquet)
    df['f'] = df['f'].astype(object)
    tm.assert_frame_equal(table_fp.to_pandas(), df)