import base64
import copy
from unittest import mock
from urllib import parse as urlparse
from oslo_utils import uuidutils
from osprofiler import _utils as osprofiler_utils
import osprofiler.profiler
from mistralclient.api import httpclient
from mistralclient.tests.unit import base
@mock.patch.object(httpclient.HTTPClient, '_get_request_options', mock.MagicMock(return_value=copy.deepcopy(EXPECTED_REQ_OPTIONS)))
def test_http_delete(self):
    m = self.requests_mock.delete(EXPECTED_URL, text='text')
    self.client.delete(API_URL)
    httpclient.HTTPClient._get_request_options.assert_called_with('delete', None)
    self.assertTrue(m.called_once)
    self.assertExpectedAuthHeaders()