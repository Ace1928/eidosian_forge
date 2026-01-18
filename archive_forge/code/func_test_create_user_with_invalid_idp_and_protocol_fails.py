import uuid
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity.shadow_users import test_backend
from keystone.tests.unit.identity.shadow_users import test_core
from keystone.tests.unit.ksfixtures import database
def test_create_user_with_invalid_idp_and_protocol_fails(self):
    baduser = unit.new_user_ref(domain_id=self.domain_id)
    baduser['federated'] = [{'idp_id': 'fakeidp', 'protocols': [{'protocol_id': 'nonexistent', 'unique_id': 'unknown'}]}]
    self.assertRaises(exception.ValidationError, self.identity_api.create_user, baduser)
    baduser['federated'][0]['idp_id'] = self.idp['id']
    self.assertRaises(exception.ValidationError, self.identity_api.create_user, baduser)