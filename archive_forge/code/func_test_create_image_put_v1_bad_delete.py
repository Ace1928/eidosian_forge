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
def test_create_image_put_v1_bad_delete(self):
    self.cloud.config.config['image_api_version'] = '1'
    args = {'name': self.image_name, 'container_format': 'bare', 'disk_format': 'qcow2', 'properties': {'owner_specified.openstack.md5': fakes.NO_MD5, 'owner_specified.openstack.sha256': fakes.NO_SHA256, 'owner_specified.openstack.object': 'images/{name}'.format(name=self.image_name), 'is_public': False}, 'validate_checksum': True}
    ret = args.copy()
    ret['id'] = self.image_id
    ret['status'] = 'success'
    self.register_uris([dict(method='GET', uri='https://image.example.com/v1/images/' + self.image_name, status_code=404), dict(method='GET', uri='https://image.example.com/v1/images/detail?name=' + self.image_name, json={'images': []}), dict(method='POST', uri='https://image.example.com/v1/images', json={'image': ret}, validate=dict(json=args)), dict(method='PUT', uri='https://image.example.com/v1/images/{id}'.format(id=self.image_id), status_code=400, validate=dict(headers={'x-image-meta-checksum': fakes.NO_MD5, 'x-glance-registry-purge-props': 'false'})), dict(method='DELETE', uri='https://image.example.com/v1/images/{id}'.format(id=self.image_id), json={'images': [ret]})])
    self.assertRaises(exceptions.HttpException, self._call_create_image, self.image_name)
    self.assert_calls()