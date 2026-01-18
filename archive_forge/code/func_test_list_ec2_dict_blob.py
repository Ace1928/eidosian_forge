import hashlib
import json
from unittest import mock
import uuid
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_db import exception as oslo_db_exception
from testtools import matchers
import urllib
from keystone.api import ec2tokens
from keystone.common import provider_api
from keystone.common import utils
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone import oauth1
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def test_list_ec2_dict_blob(self):
    """Ensure non-JSON blob data is correctly converted."""
    expected_blob, credential_id = self._create_dict_blob_credential()
    list_r = self.get('/credentials')
    list_creds = list_r.result['credentials']
    list_ids = [r['id'] for r in list_creds]
    self.assertIn(credential_id, list_ids)
    for r in list_creds:
        if r['id'] == credential_id:
            self.assertEqual(json.loads(expected_blob), json.loads(r['blob']))