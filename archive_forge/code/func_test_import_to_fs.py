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
def test_import_to_fs(self):
    import_fs = import_flow._ImportToFS(self.task.task_id, self.task_type, self.task_repo, 'http://example.com/image.qcow2')
    with mock.patch.object(script_utils, 'get_image_data_iter') as dmock:
        content = b'test'
        dmock.return_value = [content]
        with mock.patch.object(putils, 'trycmd') as tmock:
            tmock.return_value = (json.dumps({'format': 'qcow2'}), None)
            image_id = UUID1
            path = import_fs.execute(image_id)
            reader, size = glance_store.get_from_backend(path)
            self.assertEqual(4, size)
            self.assertEqual(content, b''.join(reader))
            image_path = os.path.join(self.work_dir, image_id)
            tmp_image_path = os.path.join(self.work_dir, image_path)
            self.assertTrue(os.path.exists(tmp_image_path))
            self._assert_qemu_process_limits(tmock)