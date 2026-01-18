import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_nested_api(self):
    r = self.call('nested/inner/deepfunction', _rt=bool)
    assert r is True