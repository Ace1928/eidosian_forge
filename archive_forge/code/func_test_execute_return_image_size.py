import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
@mock.patch('urllib.request')
@mock.patch('glance.async_.flows.api_image_import.json')
@mock.patch('glance.async_.utils.get_glance_endpoint')
def test_execute_return_image_size(self, mock_gge, mock_json, mock_request):
    self.config(extra_properties=['hw:numa_nodes', 'os_hash'], group='glance_download_properties')
    mock_gge.return_value = 'https://other.cloud.foo/image'
    action = self.wrapper.__enter__.return_value
    mock_json.loads.return_value = {'status': 'active', 'disk_format': 'qcow2', 'container_format': 'bare', 'hw:numa_nodes': '2', 'os_hash': 'hash', 'extra_metadata': 'hello', 'size': '12345'}
    task = import_flow._ImportMetadata(TASK_ID1, TASK_TYPE, self.context, self.wrapper, self.import_req)
    self.assertEqual(12345, task.execute())
    mock_request.Request.assert_called_once_with('https://other.cloud.foo/image/v2/images/%s' % IMAGE_ID1, headers={'X-Auth-Token': self.context.auth_token})
    mock_gge.assert_called_once_with(self.context, 'RegionTwo', 'public')
    action.set_image_attribute.assert_called_once_with(disk_format='qcow2', container_format='bare')
    action.set_image_extra_properties.assert_called_once_with({'hw:numa_nodes': '2', 'os_hash': 'hash'})