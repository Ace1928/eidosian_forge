import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_body_dict(self):
    r = self.call('bodytypes/setdict', body=({'test': 10}, {wsme.types.text: int}), _rt=int)
    self.assertEqual(r, 10)