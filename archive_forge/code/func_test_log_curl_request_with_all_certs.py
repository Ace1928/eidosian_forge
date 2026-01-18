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
def test_log_curl_request_with_all_certs(self, mock_log):
    self._test_log_curl_request_with_certs(mock_log, 'key1', 'cert1', 'cacert2')