from unittest import mock
from oslo_db import exception as db_exc
import osprofiler
import sqlalchemy
from sqlalchemy.orm import exc
import testtools
from neutron_lib.db import api as db_api
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _base
def test_multi_nested_exception_contains_deadlock(self):
    i = exceptions.MultipleExceptions([ValueError(), db_exc.DBDeadlock()])
    e = exceptions.MultipleExceptions([ValueError(), i])
    self.assertIsNone(self._decorated_function(1, e))