import hashlib
import http.client as http
import os
import subprocess
import tempfile
import time
import urllib
import uuid
import fixtures
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_serialization import jsonutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional import ft_utils as func_utils
from glance.tests import utils as test_utils
def test_hidden_images(self):
    self.api_server.show_multiple_locations = True
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(0, len(images))
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki', 'protected': False})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'os_hidden', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'os_hash_algo', 'os_hash_value', 'size', 'virtual_size', 'locations'])
    self.assertEqual(checked_keys, set(image.keys()))
    expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'os_hidden': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
    for key, value in expected_image.items():
        self.assertEqual(value, image[key], key)
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(1, len(images))
    self.assertEqual(image_id, images[0]['id'])
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-2', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki', 'os_hidden': True})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image2_id = image['id']
    checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'os_hidden', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'os_hash_algo', 'os_hash_value', 'size', 'virtual_size', 'locations'])
    self.assertEqual(checked_keys, set(image.keys()))
    expected_image = {'status': 'queued', 'name': 'image-2', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image2_id, 'protected': False, 'os_hidden': True, 'file': '/v2/images/%s/file' % image2_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
    for key, value in expected_image.items():
        self.assertEqual(value, image[key], key)
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(1, len(images))
    self.assertEqual(image_id, images[0]['id'])
    path = self._url('/v2/images?os_hidden=false')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(1, len(images))
    self.assertEqual(image_id, images[0]['id'])
    path = self._url('/v2/images?os_hidden=true')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(1, len(images))
    self.assertEqual(image2_id, images[0]['id'])
    path = self._url('/v2/images?os_hidden=abcd')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.BAD_REQUEST, response.status_code)
    path = self._url('/v2/images/%s/file' % image_id)
    headers = self._headers({'Content-Type': 'application/octet-stream'})
    image_data = b'ZZZZZ'
    response = requests.put(path, headers=headers, data=image_data)
    self.assertEqual(http.NO_CONTENT, response.status_code)
    expect_c = str(md5(image_data, usedforsecurity=False).hexdigest())
    expect_h = str(hashlib.sha512(image_data).hexdigest())
    func_utils.verify_image_hashes_and_status(self, image_id, expect_c, expect_h, size=len(image_data), status='active')
    path = self._url('/v2/images/%s/file' % image2_id)
    headers = self._headers({'Content-Type': 'application/octet-stream'})
    image_data = b'WWWWW'
    response = requests.put(path, headers=headers, data=image_data)
    self.assertEqual(http.NO_CONTENT, response.status_code)
    expect_c = str(md5(image_data, usedforsecurity=False).hexdigest())
    expect_h = str(hashlib.sha512(image_data).hexdigest())
    func_utils.verify_image_hashes_and_status(self, image2_id, expect_c, expect_h, size=len(image_data), status='active')
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/os_hidden', 'value': True}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    image = jsonutils.loads(response.text)
    self.assertTrue(image['os_hidden'])
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(0, len(images))
    path = self._url('/v2/images?os_hidden=true')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(2, len(images))
    self.assertEqual(image2_id, images[0]['id'])
    self.assertEqual(image_id, images[1]['id'])
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/os_hidden', 'value': False}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    image = jsonutils.loads(response.text)
    self.assertFalse(image['os_hidden'])
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(1, len(images))
    self.assertEqual(image_id, images[0]['id'])
    path = self._url('/v2/images/%s' % image_id)
    response = requests.delete(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images/%s' % image2_id)
    response = requests.delete(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(0, len(images))
    self.stop_servers()