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
def test_create_image_put_v2_bad_delete(self):
    self.cloud.image_api_use_tasks = False
    args = {'name': self.image_name, 'container_format': 'bare', 'disk_format': 'qcow2', 'owner_specified.openstack.md5': fakes.NO_MD5, 'owner_specified.openstack.sha256': fakes.NO_SHA256, 'owner_specified.openstack.object': 'images/{name}'.format(name=self.image_name), 'visibility': 'private'}
    ret = args.copy()
    ret['id'] = self.image_id
    ret['status'] = 'success'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('image', append=['images', self.image_name], base_url_append='v2'), status_code=404), dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2', qs_elements=['name=' + self.image_name]), validate=dict(), json={'images': []}), dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2', qs_elements=['os_hidden=True']), json={'images': []}), dict(method='POST', uri='https://image.example.com/v2/images', json=ret, validate=dict(json=args)), dict(method='PUT', uri='https://image.example.com/v2/images/{id}/file'.format(id=self.image_id), status_code=400, validate=dict(headers={'Content-Type': 'application/octet-stream'})), dict(method='DELETE', uri='https://image.example.com/v2/images/{id}'.format(id=self.image_id))])
    self.assertRaises(exceptions.HttpException, self._call_create_image, self.image_name)
    self.assert_calls()