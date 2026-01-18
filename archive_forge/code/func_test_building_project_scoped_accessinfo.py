import datetime
import uuid
from oslo_utils import timeutils
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_building_project_scoped_accessinfo(self):
    token = fixture.V3Token()
    token.set_project_scope()
    s = token.add_service(type='identity')
    s.add_standard_endpoints(public='http://url')
    token_id = uuid.uuid4().hex
    auth_ref = access.create(body=token, auth_token=token_id)
    self.assertIn('methods', auth_ref._data['token'])
    self.assertIn('catalog', auth_ref._data['token'])
    self.assertTrue(auth_ref.has_service_catalog())
    self.assertTrue(auth_ref._data['token']['catalog'])
    self.assertEqual(token_id, auth_ref.auth_token)
    self.assertEqual(token.user_name, auth_ref.username)
    self.assertEqual(token.user_id, auth_ref.user_id)
    self.assertEqual(token.role_ids, auth_ref.role_ids)
    self.assertEqual(token.role_names, auth_ref.role_names)
    self.assertIsNone(auth_ref.domain_name)
    self.assertIsNone(auth_ref.domain_id)
    self.assertEqual(token.project_name, auth_ref.project_name)
    self.assertEqual(token.project_id, auth_ref.project_id)
    self.assertEqual(auth_ref.tenant_name, auth_ref.project_name)
    self.assertEqual(auth_ref.tenant_id, auth_ref.project_id)
    self.assertEqual(token.project_domain_id, auth_ref.project_domain_id)
    self.assertEqual(token.project_domain_name, auth_ref.project_domain_name)
    self.assertEqual(token.user_domain_id, auth_ref.user_domain_id)
    self.assertEqual(token.user_domain_name, auth_ref.user_domain_name)
    self.assertFalse(auth_ref.domain_scoped)
    self.assertTrue(auth_ref.project_scoped)
    self.assertIsNone(auth_ref.project_is_domain)
    self.assertEqual(token.audit_id, auth_ref.audit_id)
    self.assertEqual(token.audit_chain_id, auth_ref.audit_chain_id)