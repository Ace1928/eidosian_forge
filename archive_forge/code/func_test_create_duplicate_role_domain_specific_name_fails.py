import uuid
from keystone.common import provider_api
from keystone.common import sql
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.assignment import test_core
from keystone.tests.unit.backend import core_sql
def test_create_duplicate_role_domain_specific_name_fails(self):
    domain = unit.new_domain_ref()
    role1 = unit.new_role_ref(domain_id=domain['id'])
    PROVIDERS.role_api.create_role(role1['id'], role1)
    role2 = unit.new_role_ref(name=role1['name'], domain_id=domain['id'])
    self.assertRaises(exception.Conflict, PROVIDERS.role_api.create_role, role2['id'], role2)