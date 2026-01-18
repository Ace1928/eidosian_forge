import io
import json
import os
from unittest import mock
import glance_store
from oslo_concurrency import processutils
from oslo_config import cfg
from glance.async_.flows import convert
from glance.async_ import taskflow_executor
from glance.common.scripts import utils as script_utils
from glance.common import utils
from glance import domain
from glance import gateway
import glance.tests.utils as test_utils
def test_convert_revert_success(self):
    image_convert = convert._Convert(self.task.task_id, self.task_type, self.img_repo)
    self.task_repo.get.return_value = self.task
    image_id = mock.sentinel.image_id
    image = mock.MagicMock(image_id=image_id, virtual_size=None)
    self.img_repo.get.return_value = image
    with mock.patch.object(processutils, 'execute') as exc_mock:
        exc_mock.return_value = ('', None)
        with mock.patch.object(os, 'remove') as rmtree_mock:
            rmtree_mock.return_value = None
            image_convert.revert(image, 'file:///tmp/test')