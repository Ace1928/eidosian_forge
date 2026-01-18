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
@mock.patch('glance.async_.flows.api_image_import.LOG')
def test_execute_confirms_lock_not_held(self, mock_log):
    wrapper = import_flow.ImportActionWrapper(self.img_repo, IMAGE_ID1, TASK_ID1)
    imagelock = import_flow._ImageLock(TASK_ID1, TASK_TYPE, wrapper)
    self.assertRaises(exception.TaskAbortedError, imagelock.execute)