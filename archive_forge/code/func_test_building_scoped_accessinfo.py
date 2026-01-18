import datetime
import uuid
from oslo_utils import timeutils
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_building_scoped_accessinfo(self):
    token = fixture.V2Token()
    token.set_scope()
    s = token.add_service('identity')
    s.add_endpoint('http://url')
    role_data = token.add_role()
    auth_ref = access.create(body=token)
    self.assertIsInstance(auth_ref, access.AccessInfoV2)
    self.assertTrue(auth_ref.has_service_catalog())
    self.assertEqual(auth_ref.auth_token, token.token_id)
    self.assertEqual(auth_ref.username, token.user_name)
    self.assertEqual(auth_ref.user_id, token.user_id)
    self.assertEqual(auth_ref.role_ids, [role_data['id']])
    self.assertEqual(auth_ref.role_names, [role_data['name']])
    self.assertEqual(auth_ref.tenant_name, token.tenant_name)
    self.assertEqual(auth_ref.tenant_id, token.tenant_id)
    self.assertEqual(auth_ref.tenant_name, auth_ref.project_name)
    self.assertEqual(auth_ref.tenant_id, auth_ref.project_id)
    self.assertIsNone(auth_ref.project_domain_id, 'default')
    self.assertIsNone(auth_ref.project_domain_name, 'Default')
    self.assertIsNone(auth_ref.user_domain_id, 'default')
    self.assertIsNone(auth_ref.user_domain_name, 'Default')
    self.assertTrue(auth_ref.project_scoped)
    self.assertFalse(auth_ref.domain_scoped)
    self.assertEqual(token.audit_id, auth_ref.audit_id)
    self.assertEqual(token.audit_chain_id, auth_ref.audit_chain_id)