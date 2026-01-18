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
def test_ec2_credential_signature_validate(self):
    """Test signature validation with a v3 ec2 credential."""
    blob, ref = unit.new_ec2_credential(user_id=self.user['id'], project_id=self.project_id)
    r = self.post('/credentials', body={'credential': ref})
    self.assertValidCredentialResponse(r, ref)
    access = blob['access'].encode('utf-8')
    self.assertEqual(hashlib.sha256(access).hexdigest(), r.result['credential']['id'])
    cred_blob = json.loads(r.result['credential']['blob'])
    self.assertEqual(blob, cred_blob)
    self._test_get_token(access=cred_blob['access'], secret=cred_blob['secret'])