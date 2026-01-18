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
@pytest.mark.skipif(bool(SKIP_PYMYSQL), reason=str(SKIP_PYMYSQL))
def test_mysql_create():
    import pymysql
    connect = pymysql.connect
    dbapi_connection = connect(host=host, user=user, password=password, database=database)
    _setup_mysql(dbapi_connection)
    _test_create(dbapi_connection)
    _setup_mysql(dbapi_connection)
    dbapi_cursor = dbapi_connection.cursor()
    _test_create(dbapi_cursor)
    dbapi_cursor.close()
    _setup_mysql(dbapi_connection)
    from sqlalchemy import create_engine
    sqlalchemy_engine = create_engine('mysql+pymysql://%s:%s@%s/%s' % (user, password, host, database))
    sqlalchemy_connection = sqlalchemy_engine.connect()
    sqlalchemy_connection.execute('SET SQL_MODE=ANSI_QUOTES')
    _test_create(sqlalchemy_connection)
    sqlalchemy_connection.close()
    _setup_mysql(dbapi_connection)
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=sqlalchemy_engine)
    sqlalchemy_session = Session()
    _test_create(sqlalchemy_session)
    sqlalchemy_session.close()