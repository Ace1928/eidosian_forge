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
def test_http_put(self):
    m = self.requests_mock.put(EXPECTED_URL, json={})
    self.client.put(API_URL, EXPECTED_BODY)
    httpclient.HTTPClient._get_request_options.assert_called_with('put', None)
    self.assertTrue(m.called_once)
    self.assertExpectedAuthHeaders()
    self.assertExpectedBody()