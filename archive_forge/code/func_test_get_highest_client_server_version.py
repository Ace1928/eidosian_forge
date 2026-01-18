import json
import logging
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as keystone_exception
from oslo_serialization import jsonutils
from cinderclient import api_versions
import cinderclient.client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@mock.patch('cinderclient.client.requests.get')
@ddt.data('3.12', '3.40')
def test_get_highest_client_server_version(self, version, mock_request):
    mock_response = utils.TestResponse({'status_code': 200, 'text': json.dumps(fakes.fake_request_get())})
    mock_request.return_value = mock_response
    url = 'http://192.168.122.127:8776/v3/e5526285ebd741b1819393f772f11fc3'
    with mock.patch.object(api_versions, 'MAX_VERSION', version):
        highest = cinderclient.client.get_highest_client_server_version(url)
    expected = version if version == '3.12' else '3.16'
    self.assertEqual(expected, highest)