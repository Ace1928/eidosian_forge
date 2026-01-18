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
@mock.patch('time.time', return_value=1400000000)
def test_generate_temp_url(self, time_mock, hmac_mock):
    hmac_mock().hexdigest.return_value = 'temp_url_signature'
    url = self.proxy.generate_temp_url(self.url, self.seconds, self.method, temp_url_key=self.key)
    key = self.key
    if not isinstance(key, bytes):
        key = key.encode('utf-8')
    self.assertEqual(url, self.expected_url)
    self.assertEqual(hmac_mock.mock_calls, [mock.call(), mock.call(key, self.expected_body, sha1), mock.call().hexdigest()])
    self.assertIsInstance(url, type(self.url))