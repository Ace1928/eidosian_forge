import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_html_format(self):
    res = self.call('argtypes/setdatetime', _accept='text/html', _no_result_decode=True)
    self.assertEqual(res.content_type, 'text/html')