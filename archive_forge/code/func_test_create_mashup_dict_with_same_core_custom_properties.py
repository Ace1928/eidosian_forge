import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def test_create_mashup_dict_with_same_core_custom_properties(self):
    image_meta = {'id': 'test-123', 'name': 'fake_image', 'status': 'active', 'created_at': '', 'min_disk': '10G', 'min_ram': '1024M', 'protected': False, 'locations': '', 'checksum': 'c1234', 'owner': '', 'disk_format': 'raw', 'container_format': 'bare', 'size': '123456789', 'virtual_size': '123456789', 'is_public': 'public', 'deleted': True, 'updated_at': '', 'properties': {'min_ram': '2048M'}}
    mashup_dict = utils.create_mashup_dict(image_meta)
    self.assertNotIn('properties', mashup_dict)
    self.assertNotEqual(image_meta['properties']['min_ram'], mashup_dict['min_ram'])
    self.assertEqual(image_meta['min_ram'], mashup_dict['min_ram'])