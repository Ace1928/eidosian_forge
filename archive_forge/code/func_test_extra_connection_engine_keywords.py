from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_extra_connection_engine_keywords(caplog, db):
    data = read_sql_table('test', db, npartitions=2, index_col='number', engine_kwargs={'echo': False}).compute()
    out = '\n'.join((r.message for r in caplog.records))
    assert out == ''
    assert_eq(data, df)
    data = read_sql_table('test', db, npartitions=2, index_col='number', engine_kwargs={'echo': True}).compute()
    out = '\n'.join((r.message for r in caplog.records))
    assert 'WHERE' in out
    assert 'FROM' in out
    assert 'SELECT' in out
    assert 'AND' in out
    assert '>= ?' in out
    assert '< ?' in out
    assert '<= ?' in out
    assert_eq(data, df)