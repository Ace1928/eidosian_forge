import base64
import copy
from unittest import mock
from urllib import parse as urlparse
from oslo_utils import uuidutils
from osprofiler import _utils as osprofiler_utils
import osprofiler.profiler
from mistralclient.api import httpclient
from mistralclient.tests.unit import base
def test_get_request_options_with_headers_for_delete(self):
    m = self.requests_mock.delete(EXPECTED_URL, text='text')
    headers = {'foo': 'bar'}
    self.client.delete(API_URL, headers=headers)
    self.assertTrue(m.called_once)
    headers = self.assertExpectedAuthHeaders()
    self.assertEqual('bar', headers['foo'])