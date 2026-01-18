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
def test_identity_headers_and_no_token_in_header(self):
    identity_headers = {'X-User-Id': 'user', 'X-Tenant-Id': 'tenant', 'X-Roles': 'roles', 'X-Identity-Status': 'Confirmed', 'X-Service-Catalog': 'service_catalog'}
    kwargs = {'token': 'fake-token', 'identity_headers': identity_headers}
    http_client_object = http.HTTPClient(self.endpoint, **kwargs)
    self.assertEqual('fake-token', http_client_object.auth_token)
    self.assertTrue(http_client_object.identity_headers.get('X-Auth-Token') is None)