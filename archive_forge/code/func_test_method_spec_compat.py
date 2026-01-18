import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_method_spec_compat(self):
    self.assertEqual(fixture.CompatArgSpec(args=['a', 'b', 'kw1'], varargs=None, keywords='kwargs', defaults=(123,)), fixture.get_method_spec(self._test_method1))
    self.assertEqual(fixture.CompatArgSpec(args=['a', 'b'], varargs='args', keywords=None, defaults=None), fixture.get_method_spec(self._test_method2))
    self.assertEqual(inspect.getfullargspec(self._test_method3), fixture.get_method_spec(self._test_method3))