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
def test_connect_retries_interval_limit(self):
    self.stub_url('GET', exc=requests.exceptions.Timeout())
    session = client_session.Session()
    retries = 20
    with mock.patch('time.sleep') as m:
        self.assertRaises(exceptions.ConnectTimeout, session.get, self.TEST_URL, connect_retries=retries)
        self.assertEqual(retries, m.call_count)
        m.assert_called_with(60.0)
    self.assertThat(self.requests_mock.request_history, matchers.HasLength(retries + 1))