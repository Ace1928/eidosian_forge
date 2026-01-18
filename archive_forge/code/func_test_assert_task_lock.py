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
def test_assert_task_lock(self):
    mock_repo = mock.MagicMock()
    mock_repo.get.return_value.extra_properties = {'os_glance_import_task': TASK_ID1}
    wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
    wrapper.assert_task_lock()
    wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, 'foo')
    self.assertRaises(exception.TaskAbortedError, wrapper.assert_task_lock)