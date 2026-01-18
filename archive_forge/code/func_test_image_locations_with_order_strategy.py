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
def test_image_locations_with_order_strategy(self):
    self.api_server.show_image_direct_url = True
    self.api_server.show_multiple_locations = True
    self.image_location_quota = 10
    self.api_server.location_strategy = 'location_order'
    preference = 'http, swift, filesystem'
    self.api_server.store_type_location_strategy_preference = preference
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'foo': 'bar', 'disk_format': 'aki', 'container_format': 'aki'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    path = self._url('/v2/images/%s' % image_id)
    headers = self._headers({'Content-Type': 'application/json'})
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    image = jsonutils.loads(response.text)
    self.assertIn('locations', image)
    self.assertEqual([], image['locations'])
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type})
    values = [{'url': 'http://127.0.0.1:%s/foo_image' % self.http_port0, 'metadata': {}}, {'url': 'http://127.0.0.1:%s/foo_image' % self.http_port1, 'metadata': {}}]
    doc = [{'op': 'replace', 'path': '/locations', 'value': values}]
    data = jsonutils.dumps(doc)
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    headers = self._headers({'Content-Type': 'application/json'})
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    image = jsonutils.loads(response.text)
    self.assertIn('locations', image)
    self.assertEqual(values, image['locations'])
    self.assertIn('direct_url', image)
    self.assertEqual(values[0]['url'], image['direct_url'])
    self.stop_servers()