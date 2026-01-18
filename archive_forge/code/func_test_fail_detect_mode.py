import logging
import os
from unittest import mock
import fixtures
from oslo_config import cfg
import sqlalchemy
from sqlalchemy.engine import base as base_engine
from sqlalchemy import exc
from sqlalchemy.pool import NullPool
from sqlalchemy import sql
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Integer, String
from sqlalchemy.orm import declarative_base
from oslo_db import exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_fail_detect_mode(self):
    log = self.useFixture(fixtures.FakeLogger(level=logging.WARN))
    mysql_conn = self.engine.raw_connection()
    self.addCleanup(mysql_conn.close)
    mysql_conn.detach()
    mysql_cursor = mysql_conn.cursor()

    def execute(statement, parameters=()):
        if "SHOW VARIABLES LIKE 'sql_mode'" in statement:
            statement = "SHOW VARIABLES LIKE 'i_dont_exist'"
        return mysql_cursor.execute(statement, parameters)
    test_engine = sqlalchemy.create_engine(self.engine.url, _initialize=False)
    with mock.patch.object(test_engine.pool, '_creator', mock.Mock(return_value=mock.Mock(cursor=mock.Mock(return_value=mock.Mock(execute=execute, fetchone=mysql_cursor.fetchone, fetchall=mysql_cursor.fetchall))))):
        engines._init_events.dispatch_on_drivername('mysql')(test_engine)
        test_engine.raw_connection()
    self.assertIn('Unable to detect effective SQL mode', log.output)