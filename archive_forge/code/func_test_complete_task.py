import io
import json
import os
from unittest import mock
import urllib
import glance_store
from oslo_concurrency import processutils as putils
from oslo_config import cfg
from taskflow import task
from taskflow.types import failure
import glance.async_.flows.base_import as import_flow
from glance.async_ import taskflow_executor
from glance.async_ import utils as async_utils
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import utils
from glance import context
from glance import domain
from glance import gateway
import glance.tests.utils as test_utils
def test_complete_task(self):
    complete_task = import_flow._CompleteTask(self.task.task_id, self.task_type, self.task_repo)
    image_id = mock.sentinel.image_id
    image = mock.MagicMock(image_id=image_id)
    self.task_repo.get.return_value = self.task
    with mock.patch.object(self.task, 'succeed') as succeed:
        complete_task.execute(image.image_id)
        succeed.assert_called_once_with({'image_id': image_id})