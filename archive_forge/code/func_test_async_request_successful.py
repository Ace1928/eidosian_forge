import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib, urlparse, parse_qsl
from libcloud.common.types import MalformedResponseError
from libcloud.common.cloudstack import CloudStackConnection
def test_async_request_successful(self):
    self.driver.path = '/async/success'
    result = self.connection._async_request('fake')
    self.assertEqual(result, {'fake': 'result'})