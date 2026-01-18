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
def test_nested_adapters(self):
    text = uuid.uuid4().hex
    token = uuid.uuid4().hex
    url = 'http://keystone.example.com/path'
    sess = client_session.Session()
    auth = CalledAuthPlugin()
    auth.ENDPOINT = url
    auth.TOKEN = token
    adap1 = adapter.Adapter(session=sess, interface='public')
    adap2 = adapter.Adapter(session=adap1, service_type='identity', auth=auth)
    self.requests_mock.get(url + '/test', text=text)
    resp = adap2.get('/test')
    self.assertEqual(text, resp.text)
    self.assertTrue(auth.get_endpoint_called)
    self.assertEqual('public', auth.endpoint_arguments['interface'])
    self.assertEqual('identity', auth.endpoint_arguments['service_type'])
    last_token = self.requests_mock.last_request.headers['X-Auth-Token']
    self.assertEqual(token, last_token)