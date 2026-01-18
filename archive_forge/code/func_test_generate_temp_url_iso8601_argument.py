from hashlib import sha1
import random
import string
import tempfile
import time
from unittest import mock
import requests_mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack.object_store.v1 import account
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit.cloud import test_object as base_test_object
from openstack.tests.unit import test_proxy_base
@mock.patch('hmac.HMAC')
def test_generate_temp_url_iso8601_argument(self, hmac_mock):
    hmac_mock().hexdigest.return_value = 'temp_url_signature'
    url = self.proxy.generate_temp_url(self.url, '2014-05-13T17:53:20Z', self.method, temp_url_key=self.key)
    self.assertEqual(url, self.expected_url)
    url = self.proxy.generate_temp_url(self.url, '2014-05-13T17:53:20Z', self.method, temp_url_key=self.key, absolute=True)
    self.assertEqual(url, self.expected_url)
    lt = time.localtime()
    expires = time.strftime(self.expires_iso8601_format[:-1], lt)
    if not isinstance(self.expected_url, str):
        expected_url = self.expected_url.replace(b'1400003600', bytes(str(int(time.mktime(lt))), encoding='ascii'))
    else:
        expected_url = self.expected_url.replace('1400003600', str(int(time.mktime(lt))))
    url = self.proxy.generate_temp_url(self.url, expires, self.method, temp_url_key=self.key)
    self.assertEqual(url, expected_url)
    expires = time.strftime(self.short_expires_iso8601_format, lt)
    lt = time.strptime(expires, self.short_expires_iso8601_format)
    if not isinstance(self.expected_url, str):
        expected_url = self.expected_url.replace(b'1400003600', bytes(str(int(time.mktime(lt))), encoding='ascii'))
    else:
        expected_url = self.expected_url.replace('1400003600', str(int(time.mktime(lt))))
    url = self.proxy.generate_temp_url(self.url, expires, self.method, temp_url_key=self.key)
    self.assertEqual(url, expected_url)