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
def test_import_flow(self):
    self.config(engine_mode='serial', group='taskflow_executor')
    img_factory = mock.MagicMock()
    executor = taskflow_executor.TaskExecutor(self.context, self.task_repo, self.img_repo, img_factory)
    self.task_repo.get.return_value = self.task

    def create_image(*args, **kwargs):
        kwargs['image_id'] = UUID1
        return self.img_factory.new_image(*args, **kwargs)
    self.img_repo.get.return_value = self.image
    img_factory.new_image.side_effect = create_image
    with mock.patch.object(script_utils, 'get_image_data_iter') as dmock:
        dmock.return_value = io.BytesIO(b'TEST_IMAGE')
        with mock.patch.object(putils, 'trycmd') as tmock:
            tmock.return_value = (json.dumps({'format': 'qcow2'}), None)
            executor.begin_processing(self.task.task_id)
            image_path = os.path.join(self.test_dir, self.image.image_id)
            tmp_image_path = os.path.join(self.work_dir, '%s.tasks_import' % image_path)
            self.assertFalse(os.path.exists(tmp_image_path))
            self.assertTrue(os.path.exists(image_path))
            self.assertEqual(1, len(list(self.image.locations)))
            self.assertEqual('file://%s%s%s' % (self.test_dir, os.sep, self.image.image_id), self.image.locations[0]['url'])
            self._assert_qemu_process_limits(tmock)