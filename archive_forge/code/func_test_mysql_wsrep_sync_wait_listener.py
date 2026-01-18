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
def test_mysql_wsrep_sync_wait_listener(self):
    with self.engine.connect() as conn:
        try:
            conn.execute(sql.text("show variables like '%wsrep_sync_wait%'")).scalars(1).one()
        except exc.NoResultFound:
            self.skipTest('wsrep_sync_wait option is not available')
    engine = self._fixture()
    with engine.connect() as conn:
        self.assertEqual('0', conn.execute(sql.text("show variables like '%wsrep_sync_wait%'")).scalars(1).one())
    for wsrep_val in (2, 1, 5):
        engine = self._fixture(mysql_wsrep_sync_wait=wsrep_val)
        with engine.connect() as conn:
            self.assertEqual(str(wsrep_val), conn.execute(sql.text("show variables like '%wsrep_sync_wait%'")).scalars(1).one())