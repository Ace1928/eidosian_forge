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
@db_test_base.backend_specific('postgresql', 'mysql')
def test_external_connection_transaction(self):
    context = oslo_context.RequestContext()
    with enginefacade.writer.connection.using(context) as connection:
        connection.execute(self.user_table.insert().values(name='u1'))
        with enginefacade.writer.independent.connection.using(context) as c2:
            self.assertIsNot(c2, connection)
            self._assert_ctx_connection(context, c2)
            c2.execute(self.user_table.insert().values(name='u2'))
            with enginefacade.writer.connection.using(context) as c3:
                self.assertIs(c2, c3)
                c3.execute(self.user_table.insert().values(name='u3'))
        self._assert_ctx_connection(context, connection)
        transaction_ctx = context.transaction_ctx
        transaction_ctx.transaction.rollback()
        transaction_ctx.transaction = connection.begin()
        connection.execute(self.user_table.insert().values(name='u4'))
    session = self.sessionmaker(autocommit=False)
    with session.begin():
        self.assertEqual([('u2',), ('u3',), ('u4',)], session.query(self.User.name).order_by(self.User.name).all())