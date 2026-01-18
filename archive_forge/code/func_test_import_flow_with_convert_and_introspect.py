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
def test_import_flow_with_convert_and_introspect(self):
    self.config(engine_mode='serial', group='taskflow_executor')
    image = self.img_factory.new_image(image_id=UUID1, disk_format='raw', container_format='bare')
    img_factory = mock.MagicMock()
    executor = taskflow_executor.TaskExecutor(self.context, self.task_repo, self.img_repo, img_factory)
    self.task_repo.get.return_value = self.task

    def create_image(*args, **kwargs):
        kwargs['image_id'] = UUID1
        return self.img_factory.new_image(*args, **kwargs)
    self.img_repo.get.return_value = image
    img_factory.new_image.side_effect = create_image
    image_path = os.path.join(self.work_dir, image.image_id)

    def fake_execute(*args, **kwargs):
        if 'info' in args:
            assert os.path.exists(args[3].split('file://')[-1])
            return (json.dumps({'virtual-size': 10737418240, 'filename': '/tmp/image.qcow2', 'cluster-size': 65536, 'format': 'qcow2', 'actual-size': 373030912, 'format-specific': {'type': 'qcow2', 'data': {'compat': '0.10'}}, 'dirty-flag': False}), None)
        open('%s.converted' % image_path, 'a').close()
        return ('', None)
    with mock.patch.object(script_utils, 'get_image_data_iter') as dmock:
        dmock.return_value = io.BytesIO(b'TEST_IMAGE')
        with mock.patch.object(processutils, 'execute') as exc_mock:
            exc_mock.side_effect = fake_execute
            executor.begin_processing(self.task.task_id)
            self.assertFalse(os.path.exists(image_path))
            self.assertEqual([], os.listdir(self.work_dir))
            self.assertEqual('qcow2', image.disk_format)
            self.assertEqual(10737418240, image.virtual_size)
            convert_call_args, _ = exc_mock.call_args_list[1]
            self.assertIn('-f', convert_call_args)