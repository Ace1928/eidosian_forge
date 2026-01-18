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
def test_logging_cacerts(self):
    path_to_certs = '/path/to/certs'
    session = client_session.Session(verify=path_to_certs)
    self.stub_url('GET', text='text')
    session.get(self.TEST_URL)
    self.assertIn('--cacert', self.logger.output)
    self.assertIn(path_to_certs, self.logger.output)