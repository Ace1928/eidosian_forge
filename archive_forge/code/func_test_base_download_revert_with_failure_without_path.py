import os
from unittest import mock
from glance_store import backend
from oslo_config import cfg
from taskflow.types import failure
from glance.async_.flows import api_image_import
import glance.common.exception
from glance import domain
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch('glance.async_.flows._internal_plugins.base_download.store_api')
def test_base_download_revert_with_failure_without_path(self, mock_store_api):
    image = self.image_repo.get.return_value
    image.status = 'importing'
    image.extra_properties['os_glance_importing_to_stores'] = 'foo'
    image.extra_properties['os_glance_failed_import'] = ''
    result = failure.Failure.from_exception(glance.common.exception.ImportTaskError())
    self.base_download_task._path = None
    self.base_download_task.revert(result)
    mock_store_api.delete_from_backend.assert_not_called()
    self.image_repo.save.assert_called_once_with(image, 'importing')
    self.assertEqual('queued', image.status)
    self.assertEqual('', image.extra_properties['os_glance_importing_to_stores'])
    self.assertEqual('foo', image.extra_properties['os_glance_failed_import'])