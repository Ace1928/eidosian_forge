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
def test_identity_headers_are_passed(self):
    identity_headers = {'X-User-Id': b'user', 'X-Tenant-Id': b'tenant', 'X-Roles': b'roles', 'X-Identity-Status': b'Confirmed', 'X-Service-Catalog': b'service_catalog'}
    kwargs = {'identity_headers': identity_headers}
    http_client = http.HTTPClient(self.endpoint, **kwargs)
    path = '/v1/images/my-image'
    self.mock.get(self.endpoint + path)
    http_client.get(path)
    headers = self.mock.last_request.headers
    for k, v in identity_headers.items():
        self.assertEqual(v, headers[k])