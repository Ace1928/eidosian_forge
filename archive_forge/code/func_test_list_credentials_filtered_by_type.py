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
def test_list_credentials_filtered_by_type(self):
    """Call ``GET  /credentials?type={type}``."""
    PROVIDERS.assignment_api.create_system_grant_for_user(self.user_id, self.role_id)
    token = self.get_system_scoped_token()
    ec2_credential = unit.new_credential_ref(user_id=uuid.uuid4().hex, project_id=self.project_id, type=CRED_TYPE_EC2)
    ec2_resp = PROVIDERS.credential_api.create_credential(ec2_credential['id'], ec2_credential)
    r = self.get('/credentials?type=cert', token=token)
    self.assertValidCredentialListResponse(r, ref=self.credential)
    for cred in r.result['credentials']:
        self.assertEqual('cert', cred['type'])
    r_ec2 = self.get('/credentials?type=ec2', token=token)
    self.assertThat(r_ec2.result['credentials'], matchers.HasLength(1))
    cred_ec2 = r_ec2.result['credentials'][0]
    self.assertValidCredentialListResponse(r_ec2, ref=ec2_resp)
    self.assertEqual(CRED_TYPE_EC2, cred_ec2['type'])
    self.assertEqual(ec2_credential['id'], cred_ec2['id'])