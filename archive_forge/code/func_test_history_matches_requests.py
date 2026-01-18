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
def test_history_matches_requests(self):
    self.setup_redirects(status_code=301)
    session = client_session.Session(redirect=True)
    req_resp = requests.get(self.REDIRECT_CHAIN[0], allow_redirects=True)
    ses_resp = session.get(self.REDIRECT_CHAIN[0])
    self.assertEqual(len(req_resp.history), len(ses_resp.history))
    for r, s in zip(req_resp.history, ses_resp.history):
        self.assertEqual(r.url, s.url)
        self.assertEqual(r.status_code, s.status_code)