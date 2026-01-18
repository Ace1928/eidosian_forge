import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
@pytest.mark.pandas
def test_compare_schemas():
    df = alltypes_sample(size=10000)
    fileh = make_sample_file(df)
    fileh2 = make_sample_file(df)
    fileh3 = make_sample_file(df[df.columns[::2]])
    assert isinstance(fileh.schema, pq.ParquetSchema)
    assert fileh.schema.equals(fileh.schema)
    assert fileh.schema == fileh.schema
    assert fileh.schema.equals(fileh2.schema)
    assert fileh.schema == fileh2.schema
    assert fileh.schema != 'arbitrary object'
    assert not fileh.schema.equals(fileh3.schema)
    assert fileh.schema != fileh3.schema
    assert isinstance(fileh.schema[0], pq.ColumnSchema)
    assert fileh.schema[0].equals(fileh.schema[0])
    assert fileh.schema[0] == fileh.schema[0]
    assert not fileh.schema[0].equals(fileh.schema[1])
    assert fileh.schema[0] != fileh.schema[1]
    assert fileh.schema[0] != 'arbitrary object'