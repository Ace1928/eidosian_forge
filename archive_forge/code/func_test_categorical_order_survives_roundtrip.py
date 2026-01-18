import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_categorical_order_survives_roundtrip():
    df = pd.DataFrame({'a': pd.Categorical(['a', 'b', 'c', 'a'], categories=['b', 'c', 'd'], ordered=True)})
    table = pa.Table.from_pandas(df)
    bos = pa.BufferOutputStream()
    pq.write_table(table, bos)
    contents = bos.getvalue()
    result = pq.read_pandas(contents).to_pandas()
    tm.assert_frame_equal(result, df)