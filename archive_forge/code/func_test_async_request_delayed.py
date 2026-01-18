import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib, urlparse, parse_qsl
from libcloud.common.types import MalformedResponseError
from libcloud.common.cloudstack import CloudStackConnection
def test_async_request_delayed(self):
    global async_delay
    self.driver.path = '/async/delayed'
    async_delay = 2
    self.connection._async_request('fake')
    self.assertEqual(async_delay, 0)