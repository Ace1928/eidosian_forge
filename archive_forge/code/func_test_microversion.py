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
def test_microversion(self):
    session = client_session.Session()
    self.stub_url('GET', text='response')
    resp = session.get(self.TEST_URL)
    self.assertTrue(resp.ok)
    self.assertRequestNotInHeader('OpenStack-API-Version')
    session = client_session.Session()
    self.stub_url('GET', text='response')
    resp = session.get(self.TEST_URL, microversion='2.30', microversion_service_type='compute', endpoint_filter={'endpoint': 'filter'})
    self.assertTrue(resp.ok)
    self.assertRequestHeaderEqual('OpenStack-API-Version', 'compute 2.30')
    self.assertRequestHeaderEqual('X-OpenStack-Nova-API-Version', '2.30')