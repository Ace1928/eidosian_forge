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
def test_json_content_type(self):
    session = client_session.Session()
    self.stub_url('POST', text='response')
    resp = session.post(self.TEST_URL, json=[{'op': 'replace', 'path': '/name', 'value': 'new_name'}], headers={'Content-Type': 'application/json-patch+json'})
    self.assertEqual('POST', self.requests_mock.last_request.method)
    self.assertEqual(resp.text, 'response')
    self.assertTrue(resp.ok)
    self.assertRequestBodyIs(json=[{'op': 'replace', 'path': '/name', 'value': 'new_name'}])
    self.assertContentTypeIs('application/json-patch+json')