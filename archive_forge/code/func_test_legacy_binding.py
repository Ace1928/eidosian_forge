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
def test_legacy_binding(self):
    key = uuid.uuid4().hex
    val = uuid.uuid4().hex
    response = json.dumps({key: val})
    self.stub_url('GET', text=response)
    auth = CalledAuthPlugin()
    sess = client_session.Session(auth=auth)
    adpt = adapter.LegacyJsonAdapter(sess, service_type=self.SERVICE_TYPE, user_agent=self.USER_AGENT)
    resp, body = adpt.get('/')
    self.assertEqual(self.SERVICE_TYPE, auth.endpoint_arguments['service_type'])
    self.assertEqual(resp.text, response)
    self.assertEqual(val, body[key])