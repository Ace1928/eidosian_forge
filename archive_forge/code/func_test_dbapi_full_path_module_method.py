from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def test_dbapi_full_path_module_method(self):
    dbapi = api.DBAPI('oslo_db.tests.test_api')
    result = dbapi.api_class_call1(1, 2, kwarg1='meow')
    expected = ((1, 2), {'kwarg1': 'meow'})
    self.assertEqual(expected, result)