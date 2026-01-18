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
def setup_redirects(self, method='GET', status_code=305, redirect_kwargs={}, final_kwargs={}):
    redirect_kwargs.setdefault('text', self.DEFAULT_REDIRECT_BODY)
    for s, d in zip(self.REDIRECT_CHAIN, self.REDIRECT_CHAIN[1:]):
        self.requests_mock.register_uri(method, s, status_code=status_code, headers={'Location': d}, **redirect_kwargs)
    final_kwargs.setdefault('status_code', 200)
    final_kwargs.setdefault('text', self.DEFAULT_RESP_BODY)
    self.requests_mock.register_uri(method, self.REDIRECT_CHAIN[-1], **final_kwargs)