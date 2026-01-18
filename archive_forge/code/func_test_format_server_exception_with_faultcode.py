import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_format_server_exception_with_faultcode(self):
    faultstring = 'boom'
    exception = Exception(faultstring)
    exception.faultcode = 'ServerError'
    ret = self._test_format_exception(exception)
    self.assertIsNone(ret['debuginfo'])
    self.assertEqual('ServerError', ret['faultcode'])
    self.assertEqual(faultstring, ret['faultstring'])