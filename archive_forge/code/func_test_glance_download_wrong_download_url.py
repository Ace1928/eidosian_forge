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
@mock.patch('glance.common.utils.validate_import_uri')
@mock.patch('glance.async_.utils.get_glance_endpoint')
def test_glance_download_wrong_download_url(self, mock_gge, mock_validate, mock_request):
    mock_validate.return_value = False
    mock_gge.return_value = 'https://other.cloud.foo/image'
    glance_download_task = glance_download._DownloadGlanceImage(self.context, self.task.task_id, self.task_type, self.action_wrapper, ['foo'], 'RegionTwo', uuidsentinel.remote_image, 'public')
    self.assertRaises(glance.common.exception.ImportTaskError, glance_download_task.execute, 12345)
    mock_request.assert_not_called()
    mock_validate.assert_called_once_with('https://other.cloud.foo/image/v2/images/%s/file' % uuidsentinel.remote_image)