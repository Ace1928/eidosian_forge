import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_http_status_retries_fixed_delay(self):
    self.stub_url('GET', status_code=409)
    session = client_session.Session()
    retries = 3
    with mock.patch('time.sleep') as m:
        self.assertRaises(exceptions.Conflict, session.get, self.TEST_URL, status_code_retries=retries, status_code_retry_delay=0.5, retriable_status_codes=[503, 409])
        self.assertEqual(retries, m.call_count)
        m.assert_has_calls([mock.call(0.5)] * retries)
    self.assertThat(self.requests_mock.request_history, matchers.HasLength(retries + 1))