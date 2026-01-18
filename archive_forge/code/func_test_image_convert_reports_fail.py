import json
import os
from unittest import mock
import glance_store
from oslo_concurrency import processutils
from oslo_config import cfg
import glance.async_.flows.api_image_import as import_flow
import glance.async_.flows.plugins.image_conversion as image_conversion
from glance.async_ import utils as async_utils
from glance.common import utils
from glance import domain
from glance import gateway
import glance.tests.utils as test_utils
def test_image_convert_reports_fail(self):
    convert = self._setup_image_convert_info_fail()
    with mock.patch.object(processutils, 'execute') as exc_mock:
        exc_mock.side_effect = [('{"format":"raw"}', ''), ('', 'some error')]
        self.assertRaises(RuntimeError, convert.execute, 'file:///test/path.raw')
        exc_mock.assert_has_calls([mock.call('qemu-img', 'info', '--output=json', '/test/path.raw', prlimit=async_utils.QEMU_IMG_PROC_LIMITS, python_exec=convert.python, log_errors=processutils.LOG_ALL_ERRORS), mock.call('qemu-img', 'convert', '-f', 'raw', '-O', 'qcow2', '/test/path.raw', '/test/path.raw.qcow2', log_errors=processutils.LOG_ALL_ERRORS)])
    self.img_repo.save.assert_not_called()