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
@mock.patch.object(os.path, 'exists')
def test_base_download_delete_staging_image_failed(self, mock_exists):
    mock_exists.return_value = True
    staging_path = 'file:///tmp/staging/temp-image'
    delete_from_fs_task = api_image_import._DeleteFromFS(self.task.task_id, self.task_type)
    with mock.patch.object(os, 'unlink') as mock_unlink:
        try:
            delete_from_fs_task.execute(staging_path)
        except OSError:
            self.assertEqual(1, mock_unlink.call_count)
        self.assertEqual(1, mock_exists.call_count)