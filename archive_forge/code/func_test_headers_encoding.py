import functools
import json
import logging
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import session
from keystoneauth1 import token_endpoint
from oslo_utils import encodeutils
import requests
from requests_mock.contrib import fixture
from urllib import parse
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
import testtools
from testtools import matchers
import types
import glanceclient
from glanceclient.common import http
from glanceclient.tests import utils
def test_headers_encoding(self):
    value = 'niño'
    fake_location = b'http://web_server:80/images/fake.img'
    headers = {'test': value, 'none-val': None, 'Name': 'value', 'x-image-meta-location': fake_location}
    encoded = http.encode_headers(headers)
    self.assertEqual(b'ni%C3%B1o', encoded[b'test'])
    self.assertNotIn('none-val', encoded)
    self.assertNotIn(b'none-val', encoded)
    self.assertEqual(b'value', encoded[b'Name'])
    self.assertEqual(fake_location, encoded[b'x-image-meta-location'])