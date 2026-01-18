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
@mock.patch('glanceclient.common.http.LOG.debug')
def test_log_curl_request_with_token_header(self, mock_log):
    fake_token = 'fake-token'
    headers = {'X-Auth-Token': fake_token}
    http_client_object = http.HTTPClient(self.endpoint, identity_headers=headers)
    http_client_object.log_curl_request('GET', '/api/v1/', headers, None, None)
    self.assertTrue(mock_log.called, 'LOG.debug never called')
    self.assertTrue(mock_log.call_args[0], 'LOG.debug called with no arguments')
    token_regex = '.*%s.*' % fake_token
    self.assertThat(mock_log.call_args[0][0], matchers.Not(matchers.MatchesRegex(token_regex)), 'token found in LOG.debug parameter')