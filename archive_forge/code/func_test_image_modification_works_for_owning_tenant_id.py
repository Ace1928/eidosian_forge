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
def test_image_modification_works_for_owning_tenant_id(self):
    rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': 'project_id:%(owner)s', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': "not ('aki':%(container_format)s and role:_member_)", 'download_image': 'role:admin or rule:restricted'}
    self.set_policy_rules(rules)
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers['content-type'] = media_type
    del headers['X-Roles']
    data = jsonutils.dumps([{'op': 'replace', 'path': '/name', 'value': 'new-name'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code)
    self.stop_servers()