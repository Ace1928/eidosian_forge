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
def test_list_credentials_filtered_by_type_and_user_id(self):
    """Call ``GET  /credentials?user_id={user_id}&type={type}``."""
    user1_id = uuid.uuid4().hex
    user2_id = uuid.uuid4().hex
    PROVIDERS.assignment_api.create_system_grant_for_user(self.user_id, self.role_id)
    token = self.get_system_scoped_token()
    credential_user1_ec2 = unit.new_credential_ref(user_id=user1_id, type=CRED_TYPE_EC2)
    credential_user1_cert = unit.new_credential_ref(user_id=user1_id)
    credential_user2_cert = unit.new_credential_ref(user_id=user2_id)
    PROVIDERS.credential_api.create_credential(credential_user1_ec2['id'], credential_user1_ec2)
    PROVIDERS.credential_api.create_credential(credential_user1_cert['id'], credential_user1_cert)
    PROVIDERS.credential_api.create_credential(credential_user2_cert['id'], credential_user2_cert)
    r = self.get('/credentials?user_id=%s&type=ec2' % user1_id, token=token)
    self.assertValidCredentialListResponse(r, ref=credential_user1_ec2)
    self.assertThat(r.result['credentials'], matchers.HasLength(1))
    cred = r.result['credentials'][0]
    self.assertEqual(CRED_TYPE_EC2, cred['type'])
    self.assertEqual(user1_id, cred['user_id'])