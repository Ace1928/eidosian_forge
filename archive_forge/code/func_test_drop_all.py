import os
from unittest import mock
from sqlalchemy.engine import url as sqla_url
from sqlalchemy import exc as sa_exc
from sqlalchemy import inspect
from sqlalchemy import schema
from sqlalchemy import types
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_drop_all(self):
    insp = inspect(self.engine)
    self.assertEqual(set(['a', 'b', 'c', 'd']), set(insp.get_table_names()))
    self._get_default_provisioned_db().backend.drop_all_objects(self.engine)
    insp = inspect(self.engine)
    self.assertEqual([], insp.get_table_names())