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
@ddt.data('http://192.168.122.127:8776/v3/e5526285ebd741b1819393f772f11fc3', 'https://192.168.122.127:8776/v3/e55285ebd741b1819393f772f11fc3', 'http://192.168.122.127/volumesv3/e5526285ebd741b1819393f772f11fc3')
def test_get_server_version(self, url, mock_request):
    mock_response = utils.TestResponse({'status_code': 200, 'text': json.dumps(fakes.fake_request_get())})
    mock_request.return_value = mock_response
    min_version, max_version = cinderclient.client.get_server_version(url)
    self.assertEqual(min_version, api_versions.APIVersion('3.0'))
    self.assertEqual(max_version, api_versions.APIVersion('3.16'))