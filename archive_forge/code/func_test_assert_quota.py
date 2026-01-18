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
def test_assert_quota(self):
    ignored = mock.MagicMock()
    task_repo = mock.MagicMock()
    task_id = 'some-task'
    enforce_fn = mock.MagicMock()
    enforce_fn.side_effect = exception.LimitExceeded
    wrapper = mock.MagicMock()
    action = wrapper.__enter__.return_value
    action.image_status = 'importing'
    self.assertRaises(exception.LimitExceeded, import_flow.assert_quota, ignored, task_repo, task_id, ['store1'], wrapper, enforce_fn)
    action.remove_importing_stores.assert_called_once_with(['store1'])
    action.set_image_attribute.assert_called_once_with(status='queued')
    task_repo.get.assert_called_once_with('some-task')
    task_repo.save.assert_called_once_with(task_repo.get.return_value)