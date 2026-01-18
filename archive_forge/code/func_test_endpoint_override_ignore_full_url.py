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
def test_endpoint_override_ignore_full_url(self):
    auth = CalledAuthPlugin()
    sess = client_session.Session(auth=auth)
    path = 'path'
    url = self.TEST_URL + path
    resp_text = uuid.uuid4().hex
    self.requests_mock.get(url, text=resp_text)
    resp = sess.get(url, endpoint_override='http://someother.url', endpoint_filter={'service_type': 'identity'})
    self.assertEqual(resp_text, resp.text)
    self.assertEqual(url, self.requests_mock.last_request.url)
    self.assertTrue(auth.get_token_called)
    self.assertFalse(auth.get_endpoint_called)
    self.assertFalse(auth.get_user_id_called)
    self.assertFalse(auth.get_project_id_called)