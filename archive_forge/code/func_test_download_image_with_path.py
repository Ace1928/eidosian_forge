import io
import operator
import tempfile
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack import connection
from openstack import exceptions
from openstack.image.v1 import image as image_v1
from openstack.image.v2 import image
from openstack.tests import fakes
from openstack.tests.unit import base
def test_download_image_with_path(self):
    self._register_image_mocks()
    output_file = tempfile.NamedTemporaryFile()
    self.cloud.download_image(self.image_name, output_path=output_file.name)
    output_file.seek(0)
    self.assertEqual(output_file.read(), self.output)
    self.assert_calls()