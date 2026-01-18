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
def test_image_convert_interpreter_configured(self):
    fake_interpreter = '/usr/bin/python2.7'
    self.config(python_interpreter=fake_interpreter, group='wsgi')
    convert = image_conversion._ConvertImage(self.context, self.task.task_id, self.task_type, self.wrapper)
    self.assertEqual(fake_interpreter, convert.python)