from unittest import mock
import urllib.error
from glance_store._drivers import filesystem
from oslo_config import cfg
from oslo_utils.fixture import uuidsentinel
from glance.async_.flows._internal_plugins import glance_download
from glance.async_.flows import api_image_import
import glance.common.exception
import glance.context
from glance import domain
import glance.tests.utils as test_utils
@mock.patch('urllib.request')
@mock.patch('glance.async_.utils.get_glance_endpoint')
def test_glance_download_no_glance_endpoint(self, mock_gge, mock_request):
    mock_gge.side_effect = glance.common.exception.GlanceEndpointNotFound(region='RegionTwo', interface='public')
    glance_download_task = glance_download._DownloadGlanceImage(self.context, self.task.task_id, self.task_type, self.action_wrapper, ['foo'], 'RegionTwo', uuidsentinel.remote_image, 'public')
    self.assertRaises(glance.common.exception.GlanceEndpointNotFound, glance_download_task.execute, 12345)
    mock_request.assert_not_called()