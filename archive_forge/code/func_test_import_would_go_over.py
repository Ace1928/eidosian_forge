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
def test_import_would_go_over(self):
    self.set_limit({'image_size_total': 5, 'image_count_total': 10, 'image_count_uploading': 10})
    self.start_server()
    image_id = self._create_and_upload(data_iter=test_utils.FakeData(3 * units.Mi))
    import_id = self._create_and_stage(data_iter=test_utils.FakeData(3 * units.Mi))
    self._import_direct(import_id, ['store1'])
    image = self._wait_for_import(import_id)
    task = self._get_latest_task(import_id)
    self.assertEqual('failure', task['status'])
    self.assertIn('image_size_total is over limit of 5 due to current usage 3 and delta 3', task['message'])
    resp = self.api_delete('/v2/images/%s' % image_id)
    self.assertEqual(204, resp.status_code)
    import_id = self._create_and_stage(data_iter=test_utils.FakeData(3 * units.Mi))
    resp = self._import_direct(import_id, ['store1'])
    self.assertEqual(202, resp.status_code)
    image = self._wait_for_import(import_id)
    self.assertEqual('active', image['status'])
    task = self._get_latest_task(import_id)
    self.assertEqual('success', task['status'])