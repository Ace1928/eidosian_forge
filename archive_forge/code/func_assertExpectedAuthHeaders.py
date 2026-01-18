import base64
import copy
from unittest import mock
from urllib import parse as urlparse
from oslo_utils import uuidutils
from osprofiler import _utils as osprofiler_utils
import osprofiler.profiler
from mistralclient.api import httpclient
from mistralclient.tests.unit import base
def assertExpectedAuthHeaders(self):
    headers = self.requests_mock.last_request.headers
    self.assertEqual(AUTH_TOKEN, headers['X-Auth-Token'])
    self.assertEqual(PROJECT_ID, headers['X-Project-Id'])
    self.assertEqual(USER_ID, headers['X-User-Id'])
    return headers