import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_move_user_between_domains(self):
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    domain2 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
    user = unit.new_user_ref(domain_id=domain1['id'])
    user = PROVIDERS.identity_api.create_user(user)
    user['domain_id'] = domain2['id']
    self.assertRaises(exception.ValidationError, PROVIDERS.identity_api.update_user, user['id'], user)