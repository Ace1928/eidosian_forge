import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_move_group_between_domains(self):
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    domain2 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
    group = unit.new_group_ref(domain_id=domain1['id'])
    group = PROVIDERS.identity_api.create_group(group)
    group['domain_id'] = domain2['id']
    self.assertRaises(exception.ValidationError, PROVIDERS.identity_api.update_group, group['id'], group)