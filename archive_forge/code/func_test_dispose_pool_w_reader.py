import collections
import contextlib
import copy
import fixtures
import pickle
import sys
from unittest import mock
import warnings
from oslo_config import cfg
from oslo_context import context as oslo_context
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines as oslo_engines
from oslo_db.sqlalchemy import orm
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db import warning
def test_dispose_pool_w_reader(self):
    facade = enginefacade.transaction_context()
    facade.configure(connection=self.engine_uri, slave_connection=self.slave_uri)
    facade.dispose_pool()
    self.assertFalse(hasattr(facade._factory, '_writer_engine'))
    self.assertFalse(hasattr(facade._factory, '_reader_engine'))
    facade._factory._start()
    facade.dispose_pool()
    self.assertEqual(facade._factory._writer_engine.pool.mock_calls, [mock.call.dispose()])
    self.assertEqual(facade._factory._reader_engine.pool.mock_calls, [mock.call.dispose()])