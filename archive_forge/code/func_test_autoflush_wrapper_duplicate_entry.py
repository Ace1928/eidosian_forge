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
def test_autoflush_wrapper_duplicate_entry(self):
    """Test a duplicate entry exception raised.

        test a duplicate entry exception raised via query.all()-> autoflush
        """
    _session = self.sessionmaker()
    with _session.begin():
        foo = self.Foo(counter=1)
        _session.add(foo)
    _session.begin()
    self.addCleanup(_session.rollback)
    foo = self.Foo(counter=1)
    _session.add(foo)
    self.assertTrue(_session.autoflush)
    self.assertRaises(exception.DBDuplicateEntry, _session.query(self.Foo).all)