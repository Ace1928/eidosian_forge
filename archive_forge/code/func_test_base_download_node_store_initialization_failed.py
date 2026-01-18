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
@mock.patch.object(cfg.ConfigOpts, 'set_override')
def test_base_download_node_store_initialization_failed(self, mock_override):
    with mock.patch.object(backend, '_load_store') as mock_load_store:
        mock_load_store.return_value = None
        self.assertRaises(glance.common.exception.BadTaskConfiguration, unit_test_utils.FakeBaseDownloadPlugin, self.task.task_id, self.task_type, self.uri, self.action_wrapper, ['foo'])
        mock_override.assert_called()