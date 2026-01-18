import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_multiindex_duplicate_values(tempdir):
    num_rows = 3
    numbers = list(range(num_rows))
    index = pd.MultiIndex.from_arrays([['foo', 'foo', 'bar'], numbers], names=['foobar', 'some_numbers'])
    df = pd.DataFrame({'numbers': numbers}, index=index)
    table = pa.Table.from_pandas(df)
    filename = tempdir / 'dup_multi_index_levels.parquet'
    _write_table(table, filename)
    result_table = _read_table(filename)
    assert table.equals(result_table)
    result_df = result_table.to_pandas()
    tm.assert_frame_equal(result_df, df)