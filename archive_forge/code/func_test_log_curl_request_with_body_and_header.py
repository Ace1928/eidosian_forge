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
@original_only
@mock.patch('glanceclient.common.http.LOG.debug')
def test_log_curl_request_with_body_and_header(self, mock_log):
    hd_name = 'header1'
    hd_val = 'value1'
    headers = {hd_name: hd_val}
    body = 'examplebody'
    self.client.log_curl_request('GET', '/api/v1/', headers, body, None)
    self.assertTrue(mock_log.called, 'LOG.debug never called')
    self.assertTrue(mock_log.call_args[0], 'LOG.debug called with no arguments')
    hd_regex = ".*\\s-H\\s+'\\s*%s\\s*:\\s*%s\\s*'.*" % (hd_name, hd_val)
    self.assertThat(mock_log.call_args[0][0], matchers.MatchesRegex(hd_regex), 'header not found in curl command')
    body_regex = ".*\\s-d\\s+'%s'\\s.*" % body
    self.assertThat(mock_log.call_args[0][0], matchers.MatchesRegex(body_regex), 'body not found in curl command')