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
def test_create_image(self):
    image_create = import_flow._CreateImage(self.task.task_id, self.task_type, self.task_repo, self.img_repo, self.img_factory)
    self.task_repo.get.return_value = self.task
    with mock.patch.object(image_import, 'create_image') as ci_mock:
        ci_mock.return_value = mock.Mock()
        image_create.execute()
        ci_mock.assert_called_once_with(self.img_repo, self.img_factory, {'container_format': 'bare', 'disk_format': 'qcow2'}, self.task.task_id)