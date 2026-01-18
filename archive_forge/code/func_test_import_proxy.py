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
def test_import_proxy(self):
    resp = requests.Response()
    resp.status_code = 202
    resp.headers['x-openstack-request-id'] = 'req-remote'
    self.ksa_client.return_value.post.return_value = resp
    self.config(worker_self_reference_url='http://worker1')
    self.start_server(set_worker_url=False)
    image_id = self._create_and_stage()
    image = self.api_get('/v2/images/%s' % image_id).json
    self.assertIn('container_format', image)
    self.assertNotIn('os_glance_stage_host', image)
    self.config(worker_self_reference_url='http://worker2')
    self.start_server(set_worker_url=False)
    r = self._import_direct(image_id, ['store1'])
    self.assertEqual(202, r.status_code)
    self.assertEqual('req-remote', r.headers['x-openstack-request-id'])
    self.ksa_client.return_value.post.assert_called_once_with('http://worker1/v2/images/%s/import' % image_id, timeout=60, json={'method': {'name': 'glance-direct'}, 'stores': ['store1'], 'all_stores': False})