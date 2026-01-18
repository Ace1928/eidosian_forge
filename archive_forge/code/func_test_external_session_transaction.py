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
def test_external_session_transaction(self):
    context = oslo_context.RequestContext()
    with enginefacade.writer.using(context) as session:
        session.add(self.User(name='u1'))
        session.flush()
        with enginefacade.writer.independent.using(context) as s2:
            self.assertIsNot(s2, session)
            self._assert_ctx_session(context, s2)
            s2.add(self.User(name='u2'))
            with enginefacade.writer.using(context) as s3:
                self.assertIs(s3, s2)
                s3.add(self.User(name='u3'))
        self._assert_ctx_session(context, session)
        session.rollback()
        session.begin()
        session.add(self.User(name='u4'))
    session = self.sessionmaker(autocommit=False)
    with session.begin():
        self.assertEqual([('u2',), ('u3',), ('u4',)], session.query(self.User.name).order_by(self.User.name).all())