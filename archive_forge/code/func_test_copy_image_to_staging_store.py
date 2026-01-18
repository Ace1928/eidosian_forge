import datetime
import os
from unittest import mock
import glance_store as store_api
from oslo_config import cfg
from glance.async_.flows._internal_plugins import copy_image
from glance.async_.flows import api_image_import
import glance.common.exception as exception
from glance import domain
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch.object(store_api, 'get_store_from_store_identifier')
def test_copy_image_to_staging_store(self, mock_store_api):
    mock_store_api.return_value = self.staging_store
    copy_image_task = copy_image._CopyImage(self.task.task_id, self.task_type, self.image_repo, self.action_wrapper)
    with mock.patch.object(self.image_repo, 'get') as get_mock:
        get_mock.return_value = mock.MagicMock(image_id=self.images[0]['id'], locations=self.images[0]['locations'], extra_properties={'os_glance_import_task': self.task.task_id}, status=self.images[0]['status'])
        with mock.patch.object(store_api, 'get') as get_data:
            get_data.return_value = (b'dddd', 4)
            copy_image_task.execute()
            self.staging_store.add.assert_called_once()
            mock_store_api.assert_called_once_with('os_glance_staging_store')