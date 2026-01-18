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
def test_expired_token_has_changed(self):
    fake_token = b'fake-token'
    http_client = http.HTTPClient(self.endpoint, token=fake_token)
    path = '/v1/images/my-image'
    self.mock.get(self.endpoint + path)
    http_client.get(path)
    headers = self.mock.last_request.headers
    self.assertEqual(fake_token, headers['X-Auth-Token'])
    refreshed_token = b'refreshed-token'
    http_client.auth_token = refreshed_token
    http_client.get(path)
    headers = self.mock.last_request.headers
    self.assertEqual(refreshed_token, headers['X-Auth-Token'])
    unicode_token = 'niño+=='
    http_client.auth_token = unicode_token
    http_client.get(path)
    headers = self.mock.last_request.headers
    self.assertEqual(b'ni%C3%B1o+==', headers['X-Auth-Token'])