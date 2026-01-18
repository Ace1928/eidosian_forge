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
def test_context_deepcopy_on_connection(self):
    context = oslo_context.RequestContext()
    with enginefacade.writer.connection.using(context) as conn:
        ctx2 = copy.deepcopy(context)
        self._assert_ctx_connection(ctx2, conn)
        with enginefacade.writer.connection.using(ctx2) as conn2:
            self.assertIs(conn, conn2)
            self._assert_ctx_connection(ctx2, conn2)
            conn2.execute(self.user_table.insert().values(name='u1'))
        self._assert_ctx_connection(ctx2, conn2)
    session = self.sessionmaker(autocommit=False)
    with session.begin():
        self.assertEqual('u1', session.query(self.User.name).scalar())