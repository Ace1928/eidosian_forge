import uuid
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity.shadow_users import test_backend
from keystone.tests.unit.identity.shadow_users import test_core
from keystone.tests.unit.ksfixtures import database
def test_get_user_when_user_has_federated_object(self):
    fed_dict = unit.new_federated_user_ref(idp_id=self.idp['id'], protocol_id=self.protocol['id'])
    user = self.shadow_users_api.create_federated_user(self.domain_id, fed_dict)
    user_ref = self.identity_api.get_user(user['id'])
    self.assertIn('federated', user_ref)
    self.assertEqual(1, len(user_ref['federated']))
    self.assertFederatedDictsEqual(fed_dict, user_ref['federated'][0])