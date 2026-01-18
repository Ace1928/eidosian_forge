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
def test_image_import_qcow_virtual_size_calculation(self):
    self.start_servers(**self.__dict__.copy())
    headers = self._headers({'Content-Type': 'application/json'})
    data = jsonutils.dumps({'name': 'myqcow', 'disk_format': 'qcow2', 'container_format': 'bare'})
    response = requests.post(self._url('/v2/images'), headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code, 'Failed to create: %s' % response.text)
    image = response.json()
    fn = self._create_qcow(128 * units.Mi)
    raw_size = os.path.getsize(fn)
    headers = self._headers({'Content-Type': 'application/octet-stream'})
    response = requests.put(self._url('/v2/images/%s/stage' % image['id']), headers=headers, data=open(fn, 'rb').read())
    os.remove(fn)
    self.assertEqual(http.NO_CONTENT, response.status_code)
    func_utils.verify_image_hashes_and_status(self, image['id'], status='uploading', size=raw_size)
    path = self._url('/v2/images/%s/import' % image['id'])
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    data = jsonutils.dumps({'method': {'name': 'glance-direct'}})
    response = requests.post(self._url('/v2/images/%s/import' % image['id']), headers=headers, data=data)
    self.assertEqual(http.ACCEPTED, response.status_code)
    path = self._url('/v2/images/%s' % image['id'])
    func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=15, delay_sec=0.2)
    response = requests.get(self._url('/v2/images/%s' % image['id']), headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    image = response.json()
    self.assertEqual(128 * units.Mi, image['virtual_size'])
    self.assertEqual(raw_size, image['size'])