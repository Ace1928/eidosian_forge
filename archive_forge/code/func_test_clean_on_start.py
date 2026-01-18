import http.client as http
import os
import socket
import time
from oslo_serialization import jsonutils
from oslo_utils.fixture import uuidsentinel as uuids
import requests
from glance.common import wsgi
from glance.tests import functional
def test_clean_on_start(self):
    staging = os.path.join(self.test_dir, 'staging')
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    path = self._url('/v2/images/%s/stage' % image_id)
    headers = self._headers({'Content-Type': 'application/octet-stream'})
    image_data = b'ZZZZZ'
    response = requests.put(path, headers=headers, data=image_data)
    self.assertEqual(http.NO_CONTENT, response.status_code)
    self.my_api_server.stop()
    open(os.path.join(staging, 'foo'), 'w')
    open(os.path.join(staging, uuids.stale), 'w')
    open(os.path.join(staging, uuids.converting), 'w')
    converting_fn = os.path.join(staging, '%s.qcow2' % uuids.converting)
    decompressing_fn = os.path.join(staging, '%s.uc' % uuids.decompressing)
    open(converting_fn, 'w')
    open(decompressing_fn, 'w')
    self.my_api_server.needs_database = False
    self.start_with_retry(self.my_api_server, 'api_port', 3, **self.__dict__.copy())
    for i in range(0, 10):
        try:
            requests.get(self._url('/v2/images'))
        except Exception:
            pass
        else:
            files = os.listdir(staging)
            if len(files) == 2:
                break
        time.sleep(1)
    self.assertTrue(os.path.exists(os.path.join(staging, 'foo')))
    self.assertTrue(os.path.exists(os.path.join(staging, image_id)))
    self.assertFalse(os.path.exists(os.path.join(staging, uuids.stale)))
    self.assertFalse(os.path.exists(converting_fn))
    self.assertFalse(os.path.exists(decompressing_fn))
    self.stop_servers()