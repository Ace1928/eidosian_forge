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
def test_import_proxy_fail_on_remote(self):
    resp = requests.Response()
    resp.url = '/v2'
    resp.status_code = 409
    resp.reason = 'Something Failed (tm)'
    self.ksa_client.return_value.post.return_value = resp
    self.ksa_client.return_value.delete.return_value = resp
    self.config(worker_self_reference_url='http://worker1')
    self.start_server(set_worker_url=False)
    image_id = self._create_and_stage()
    self.config(worker_self_reference_url='http://worker2')
    self.start_server(set_worker_url=False)
    r = self._import_direct(image_id, ['store1'])
    self.assertEqual(409, r.status_code)
    self.assertEqual('409 Something Failed (tm)', r.status)
    r = self.api_delete('/v2/images/%s' % image_id)
    self.assertEqual(409, r.status_code)
    self.assertEqual('409 Something Failed (tm)', r.status)