import datetime
import uuid
from oslo_utils import timeutils
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_building_unscoped_accessinfo(self):
    token = fixture.V3Token()
    token_id = uuid.uuid4().hex
    auth_ref = access.create(body=token, auth_token=token_id)
    self.assertIn('methods', auth_ref._data['token'])
    self.assertFalse(auth_ref.has_service_catalog())
    self.assertNotIn('catalog', auth_ref._data['token'])
    self.assertEqual(token_id, auth_ref.auth_token)
    self.assertEqual(token.user_name, auth_ref.username)
    self.assertEqual(token.user_id, auth_ref.user_id)
    self.assertEqual(auth_ref.role_ids, [])
    self.assertEqual(auth_ref.role_names, [])
    self.assertIsNone(auth_ref.project_name)
    self.assertIsNone(auth_ref.project_id)
    self.assertFalse(auth_ref.domain_scoped)
    self.assertFalse(auth_ref.project_scoped)
    self.assertIsNone(auth_ref.project_is_domain)
    self.assertEqual(token.user_domain_id, auth_ref.user_domain_id)
    self.assertEqual(token.user_domain_name, auth_ref.user_domain_name)
    self.assertIsNone(auth_ref.project_domain_id)
    self.assertIsNone(auth_ref.project_domain_name)
    self.assertEqual(auth_ref.expires, timeutils.parse_isotime(token['token']['expires_at']))
    self.assertEqual(auth_ref.issued, timeutils.parse_isotime(token['token']['issued_at']))
    self.assertEqual(auth_ref.expires, token.expires)
    self.assertEqual(auth_ref.issued, token.issued)
    self.assertEqual(auth_ref.audit_id, token.audit_id)
    self.assertIsNone(auth_ref.audit_chain_id)
    self.assertIsNone(token.audit_chain_id)
    self.assertIsNone(auth_ref.bind)