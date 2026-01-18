import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def setup_mock(versions_in):
    jsondata = {'versions': [dict({'status': discover.Status.CURRENT, 'id': 'v2.2', 'links': [{'href': V3_URL, 'rel': 'self'}]}, **versions_in)]}
    self.requests_mock.get(V3_URL, status_code=200, json=jsondata)