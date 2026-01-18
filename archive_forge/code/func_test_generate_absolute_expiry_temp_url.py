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
@mock.patch('hmac.HMAC.hexdigest', return_value='temp_url_signature')
def test_generate_absolute_expiry_temp_url(self, hmac_mock):
    if isinstance(self.expected_url, bytes):
        expected_url = self.expected_url.replace(b'1400003600', b'2146636800')
    else:
        expected_url = self.expected_url.replace(u'1400003600', u'2146636800')
    url = self.proxy.generate_temp_url(self.url, 2146636800, self.method, absolute=True, temp_url_key=self.key)
    self.assertEqual(url, expected_url)