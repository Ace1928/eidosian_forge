from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_division_or_partition(db):
    with pytest.raises(TypeError, match="either 'divisions' or 'npartitions'"):
        read_sql_table('test', db, columns=['name'], index_col='number', divisions=[0, 2, 4], npartitions=3)
    out = read_sql_table('test', db, index_col='number', bytes_per_chunk=100)
    m = out.memory_usage_per_partition(deep=True, index=True).compute()
    assert (50 < m).all() and (m < 200).all()
    assert_eq(out, df)