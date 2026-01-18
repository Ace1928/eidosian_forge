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
def test_base_download_revert_without_failure_multi_store(self, mock_store_api):
    enabled_backends = {'fast': 'file', 'cheap': 'file'}
    self.config(enabled_backends=enabled_backends)
    self.base_download_task.revert('/path/to_downloaded_data')
    mock_store_api.delete.assert_called_once_with('/path/to_downloaded_data', None)