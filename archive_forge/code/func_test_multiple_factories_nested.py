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
def test_multiple_factories_nested(self):
    """Test that the instrumentation applied to a context class supports

        nested calls among multiple _TransactionContextManager objects.

        """
    mgr1 = enginefacade.transaction_context()
    mgr1.configure(connection=self.engine_uri, slave_connection=self.slave_uri)
    mgr2 = enginefacade.transaction_context()
    mgr2.configure(connection=self.engine_uri, slave_connection=self.slave_uri)
    context = oslo_context.RequestContext()
    with mgr1.writer.using(context):
        self.assertIs(context.transaction_ctx.factory, mgr1._factory)
        self.assertIsNot(context.transaction_ctx.factory, mgr2._factory)
        with mgr2.writer.using(context):
            self.assertIsNot(context.transaction_ctx.factory, mgr1._factory)
            self.assertIs(context.transaction_ctx.factory, mgr2._factory)
            self.assertIsNotNone(context.session)
        self.assertIs(context.transaction_ctx.factory, mgr1._factory)
        self.assertIsNot(context.transaction_ctx.factory, mgr2._factory)
        self.assertIsNotNone(context.session)
    self.assertRaises(exception.NoEngineContextEstablished, getattr, context, 'transaction_ctx')