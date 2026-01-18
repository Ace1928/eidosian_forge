from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
@mock.patch.object(filecache, 'save_data', autospec=True)
@mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
def test_negotiate_version_server_newer(self, mock_pvh, mock_save_data):
    mock_pvh.return_value = ('1.1', '1.10')
    mock_conn = mock.MagicMock()
    result = self.test_object.negotiate_version(mock_conn, self.response)
    self.assertEqual('1.6', result)
    self.assertEqual(1, mock_pvh.call_count)
    mock_save_data.assert_called_once_with(host=DEFAULT_HOST, port=DEFAULT_PORT, data='1.6')