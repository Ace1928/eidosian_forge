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
def test_logger_object_passed(self):
    logger = logging.getLogger(uuid.uuid4().hex)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    string_io = io.StringIO()
    handler = logging.StreamHandler(string_io)
    logger.addHandler(handler)
    auth = AuthPlugin()
    sess = client_session.Session(auth=auth)
    adpt = adapter.Adapter(sess, auth=auth, logger=logger)
    response = {uuid.uuid4().hex: uuid.uuid4().hex}
    self.stub_url('GET', json=response, headers={'Content-Type': 'application/json'})
    resp = adpt.get(self.TEST_URL, logger=logger)
    self.assertEqual(response, resp.json())
    output = string_io.getvalue()
    self.assertIn(self.TEST_URL, output)
    self.assertIn(list(response.keys())[0], output)
    self.assertIn(list(response.values())[0], output)
    self.assertNotIn(list(response.keys())[0], self.logger.output)
    self.assertNotIn(list(response.values())[0], self.logger.output)