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
def test_app_cred_ec2_credential(self):
    """Test creating ec2 credential from an application credential.

        Call ``POST /credentials``.
        """
    ref = unit.new_application_credential_ref(roles=[{'id': self.role_id}])
    del ref['id']
    r = self.post('/users/%s/application_credentials' % self.user_id, body={'application_credential': ref})
    app_cred = r.result['application_credential']
    auth_data = self.build_authentication_request(app_cred_id=app_cred['id'], secret=app_cred['secret'])
    r = self.v3_create_token(auth_data)
    token_id = r.headers.get('X-Subject-Token')
    blob, ref = unit.new_ec2_credential(user_id=self.user_id, project_id=self.project_id)
    r = self.post('/credentials', body={'credential': ref}, token=token_id)
    ret_ref = ref.copy()
    ret_blob = blob.copy()
    ret_blob['app_cred_id'] = app_cred['id']
    ret_ref['blob'] = json.dumps(ret_blob)
    self.assertValidCredentialResponse(r, ref=ret_ref)
    access = blob['access'].encode('utf-8')
    self.assertEqual(hashlib.sha256(access).hexdigest(), r.result['credential']['id'])
    role = unit.new_role_ref(name='reader')
    role_id = role['id']
    PROVIDERS.role_api.create_role(role_id, role)
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_id, self.project_id, role_id)
    ret_blob = json.loads(r.result['credential']['blob'])
    ec2token = self._test_get_token(access=ret_blob['access'], secret=ret_blob['secret'])
    ec2_roles = [role['id'] for role in ec2token['roles']]
    self.assertIn(self.role_id, ec2_roles)
    self.assertNotIn(role_id, ec2_roles)
    self.post('/credentials', body={'credential': ref}, token=token_id, expected_status=http.client.CONFLICT)