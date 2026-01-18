import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_filters_inclusive_datetime(tempdir):
    path = tempdir / 'timestamps.parquet'
    pd.DataFrame({'dates': pd.date_range('2020-01-01', periods=10, freq='D'), 'id': range(10)}).to_parquet(path, use_deprecated_int96_timestamps=True)
    table = pq.read_table(path, filters=[('dates', '<=', datetime.datetime(2020, 1, 5))])
    assert table.column('id').to_pylist() == [0, 1, 2, 3, 4]