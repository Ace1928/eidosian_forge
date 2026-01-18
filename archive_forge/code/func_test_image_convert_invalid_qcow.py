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
def test_image_convert_invalid_qcow(self):
    data = {'format': 'qcow2', 'backing-filename': '/etc/hosts'}
    convert = self._setup_image_convert_info_fail()
    with mock.patch.object(processutils, 'execute') as exc_mock:
        exc_mock.return_value = (json.dumps(data), '')
        e = self.assertRaises(RuntimeError, convert.execute, 'file:///test/path.qcow')
        self.assertEqual('QCOW images with backing files are not allowed', str(e))