import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_missing_argument(self):
    try:
        r = self.call('argtypes/setdatetime')
        print(r)
        assert 'No error raised'
    except CallException as e:
        self.assertEqual(e.faultcode, 'Client')
        self.assertEqual(e.faultstring, 'Missing argument: "value"')