import contextlib
import itertools
from unittest import mock
import sqlalchemy as sqla
from sqlalchemy import event
import sqlalchemy.exc
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy import sql
from oslo_db import exception
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import exc_filters
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db.tests import utils as test_utils
def test_cause_for_failed_flush_plus_no_savepoint(self):
    session = self.sessionmaker()
    with session.begin():
        session.add(self.A(id=1))
    try:
        with session.begin():
            try:
                with session.begin_nested():
                    session.execute(sql.text('rollback'))
                    session.add(self.A(id=1))
            except exception.DBError as dbe_inner:
                self.assertIsInstance(dbe_inner.cause, exception.DBDuplicateEntry)
    except exception.DBError as dbe_outer:
        self.AssertIsInstance(dbe_outer.cause, exception.DBDuplicateEntry)
    try:
        with session.begin():
            session.add(self.A(id=1))
    except exception.DBError as dbe_outer:
        self.assertIsNone(dbe_outer.cause)