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
def test_create_image_volume(self):
    self.register_uris([self.get_cinder_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('volumev3', append=['volumes', self.volume_id, 'action']), json={'os-volume_upload_image': {'image_id': self.image_id}}, validate=dict(json={'os-volume_upload_image': {'container_format': 'bare', 'disk_format': 'qcow2', 'force': False, 'image_name': 'fake_image'}})), self.get_glance_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2'), json=self.fake_search_return)])
    self.cloud.create_image('fake_image', self.imagefile.name, wait=True, timeout=1, volume={'id': self.volume_id})
    self.assert_calls()