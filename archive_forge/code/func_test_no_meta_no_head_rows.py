from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_no_meta_no_head_rows(db):
    with pytest.raises(ValueError):
        read_sql_table('test', db, index_col='number', head_rows=0, npartitions=1)