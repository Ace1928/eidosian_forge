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
def test_download_image_raises_service_unavailable(self):
    """Test image download returns HTTPServiceUnavailable."""
    self.api_server.show_multiple_locations = True
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type})
    thread, httpd, http_port = test_utils.start_http_server(image_id, 'image-1')
    values = [{'url': 'http://127.0.0.1:%s/image-1' % http_port, 'metadata': {'idx': '0'}}]
    doc = [{'op': 'replace', 'path': '/locations', 'value': values}]
    data = jsonutils.dumps(doc)
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code)
    path = self._url('/v2/images/%s/file' % image_id)
    headers = self._headers({'Content-Type': 'application/json'})
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    httpd.shutdown()
    httpd.server_close()
    path = self._url('/v2/images/%s/file' % image_id)
    headers = self._headers({'Content-Type': 'application/json'})
    response = requests.get(path, headers=headers)
    self.assertEqual(http.SERVICE_UNAVAILABLE, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.delete(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.NOT_FOUND, response.status_code)
    self.stop_servers()