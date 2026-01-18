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
@mock.patch.object(http.VersionNegotiationMixin, '_make_simple_request', autospec=True)
@mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
def test_negotiate_version_server_user_list_single_value(self, mock_pvh, mock_msr, mock_save_data):
    mock_pvh.side_effect = iter([(None, None), ('1.1', '1.26')])
    mock_conn = mock.MagicMock()
    self.test_object.api_version_select_state = 'user'
    self.test_object.os_ironic_api_version = ['1.01']
    result = self.test_object.negotiate_version(mock_conn, None)
    self.assertEqual('1.1', result)
    self.assertEqual('negotiated', self.test_object.api_version_select_state)
    self.assertEqual('1.1', self.test_object.os_ironic_api_version)
    self.assertTrue(mock_msr.called)
    self.assertEqual(2, mock_pvh.call_count)
    self.assertEqual(1, mock_save_data.call_count)