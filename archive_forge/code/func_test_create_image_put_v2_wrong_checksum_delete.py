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
def test_create_image_put_v2_wrong_checksum_delete(self):
    self.cloud.image_api_use_tasks = False
    fake_image = self.fake_image_dict
    fake_image['owner_specified.openstack.md5'] = 'a'
    fake_image['owner_specified.openstack.sha256'] = 'b'
    self.register_uris([dict(method='POST', uri=self.get_mock_url('image', append=['images'], base_url_append='v2'), json=self.fake_image_dict, validate=dict(json={'container_format': 'bare', 'disk_format': 'qcow2', 'name': self.image_name, 'owner_specified.openstack.md5': fake_image['owner_specified.openstack.md5'], 'owner_specified.openstack.object': self.object_name, 'owner_specified.openstack.sha256': fake_image['owner_specified.openstack.sha256'], 'visibility': 'private'})), dict(method='PUT', uri=self.get_mock_url('image', append=['images', self.image_id, 'file'], base_url_append='v2'), request_headers={'Content-Type': 'application/octet-stream'}), dict(method='GET', uri=self.get_mock_url('image', append=['images', self.fake_image_dict['id']], base_url_append='v2'), json=fake_image), dict(method='DELETE', uri='https://image.example.com/v2/images/{id}'.format(id=self.image_id))])
    self.assertRaises(exceptions.SDKException, self.cloud.create_image, self.image_name, self.imagefile.name, is_public=False, md5='a', sha256='b', allow_duplicates=True, validate_checksum=True)
    self.assert_calls()