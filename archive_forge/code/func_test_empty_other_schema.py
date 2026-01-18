from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
@pytest.mark.skip(reason='Requires a postgres server. Sqlite does not support multiple schemas.')
def test_empty_other_schema():
    from sqlalchemy import DDL, Column, Integer, MetaData, Table, create_engine, event
    pg_host = 'localhost'
    pg_port = '5432'
    pg_user = 'user'
    pg_pass = 'pass'
    pg_db = 'db'
    db_url = f'postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}'
    table_name = 'empty_table'
    schema_name = 'other_schema'
    engine = create_engine(db_url)
    metadata = MetaData()
    table = Table(table_name, metadata, Column('id', Integer, primary_key=True), Column('col2', Integer), schema=schema_name)
    event.listen(metadata, 'before_create', DDL('CREATE SCHEMA IF NOT EXISTS %s' % schema_name))
    metadata.create_all(engine)
    dask_df = read_sql_table(table.name, db_url, index_col='id', schema=table.schema, npartitions=1)
    assert dask_df.index.name == 'id'
    assert dask_df.col2.dtype == np.dtype('int64')
    pd_dataframe = dask_df.compute()
    assert pd_dataframe.empty is True
    engine.execute('DROP SCHEMA IF EXISTS %s CASCADE' % schema_name)