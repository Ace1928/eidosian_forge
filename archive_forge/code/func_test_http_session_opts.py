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
def test_http_session_opts(self):
    session = client_session.Session(cert='cert.pem', timeout=5, verify='certs')
    FAKE_RESP = utils.TestResponse({'status_code': 200, 'text': 'resp'})
    RESP = mock.Mock(return_value=FAKE_RESP)
    with mock.patch.object(session.session, 'request', RESP) as mocked:
        session.post(self.TEST_URL, data='value')
        mock_args, mock_kwargs = mocked.call_args
        self.assertEqual(mock_args[0], 'POST')
        self.assertEqual(mock_args[1], self.TEST_URL)
        self.assertEqual(mock_kwargs['data'], 'value')
        self.assertEqual(mock_kwargs['cert'], 'cert.pem')
        self.assertEqual(mock_kwargs['verify'], 'certs')
        self.assertEqual(mock_kwargs['timeout'], 5)