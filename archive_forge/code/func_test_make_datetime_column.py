from __future__ import absolute_import, print_function, division
import logging
from datetime import datetime, date
import sqlite3
import pytest
from petl.io.db import fromdb, todb
from petl.io.db_create import make_sqlalchemy_column
from petl.test.helpers import ieq, eq_
from petl.util.vis import look
from petl.test.io.test_db_server import user, password, host, database
def test_make_datetime_column():
    sql_col = make_sqlalchemy_column([datetime(2014, 1, 1, 1, 1, 1, 1), datetime(2014, 1, 1, 1, 1, 1, 2)], 'name')
    expect = Column('name', DateTime(), nullable=False)
    eq_(str(expect.type), str(sql_col.type))