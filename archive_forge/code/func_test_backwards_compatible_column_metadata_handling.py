import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_backwards_compatible_column_metadata_handling(datadir):
    if Version('2.2.0') <= Version(pd.__version__):
        pytest.skip('Regression in pandas 2.2.0')
    expected = pd.DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3], 'c': pd.date_range('2017-01-01', periods=3, tz='Europe/Brussels')})
    expected.index = pd.MultiIndex.from_arrays([['a', 'b', 'c'], pd.date_range('2017-01-01', periods=3, tz='Europe/Brussels')], names=['index', None])
    path = datadir / 'v0.7.1.column-metadata-handling.parquet'
    table = _read_table(path)
    result = table.to_pandas()
    tm.assert_frame_equal(result, expected)
    table = _read_table(path, columns=['a'])
    result = table.to_pandas()
    tm.assert_frame_equal(result, expected[['a']].reset_index(drop=True))