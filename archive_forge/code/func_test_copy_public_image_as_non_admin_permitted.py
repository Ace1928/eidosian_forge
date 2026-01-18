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
def test_copy_public_image_as_non_admin_permitted(self):
    rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': '', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': '', 'download_image': '', 'add_member': '', 'publicize_image': '', 'copy_image': "'public':%(visibility)s"}
    self.set_policy_rules(rules)
    image_id, response = self._test_copy_public_image_as_non_admin()
    self.assertEqual(http.ACCEPTED, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    func_utils.wait_for_copying(request_path=path, request_headers=self._headers(), stores=['file2'], max_sec=40, delay_sec=0.2, start_delay_sec=1)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    self.assertIn('file2', jsonutils.loads(response.text)['stores'])