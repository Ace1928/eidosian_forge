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
def test_log_curl_request_with_non_ascii_char(self):
    try:
        headers = {'header1': 'value1¥¦'}
        body = 'examplebody¥¦'
        self.client.log_curl_request('GET', '/api/v1/¥', headers, body, None)
    except UnicodeDecodeError as e:
        self.fail("Unexpected UnicodeDecodeError exception '%s'" % e)