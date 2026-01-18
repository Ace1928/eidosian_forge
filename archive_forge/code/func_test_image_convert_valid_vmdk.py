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
def test_image_convert_valid_vmdk(self):
    with mock.patch.object(CONF.image_format, 'vmdk_allowed_types', new=['monolithicSparse', 'monolithicFlat']):
        self.assertRaises(FileNotFoundError, self._test_image_convert_invalid_vmdk)