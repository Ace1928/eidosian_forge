import datetime
import uuid
from oslo_utils import timeutils
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_grizzly_token(self):
    grizzly_token = {'access': {'token': {'id': uuid.uuid4().hex, 'expires': '2020-01-01T00:00:10.000123Z'}, 'user': {'id': 'user_id1', 'name': 'user_name1', 'tenantId': 'tenant_id1', 'tenantName': 'tenant_name1', 'roles': [{'name': 'role1'}, {'name': 'role2'}]}}}
    auth_ref = access.create(body=grizzly_token)
    self.assertIsInstance(auth_ref, access.AccessInfoV2)
    self.assertEqual(auth_ref.project_id, 'tenant_id1')
    self.assertEqual(auth_ref.project_name, 'tenant_name1')
    self.assertIsNone(auth_ref.project_domain_id)
    self.assertIsNone(auth_ref.project_domain_name)
    self.assertIsNone(auth_ref.user_domain_id, 'default')
    self.assertIsNone(auth_ref.user_domain_name, 'Default')
    self.assertEqual(auth_ref.role_names, ['role1', 'role2'])