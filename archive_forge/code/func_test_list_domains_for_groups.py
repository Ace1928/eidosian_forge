from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_domains_for_groups(self):
    """Test retrieving domains for a list of groups.

        Test Plan:

        - Create three domains, three groups and one role
        - Assign a non-inherited group role to two domains, and an inherited
          group role to the third
        - Ensure only the domains with non-inherited roles are returned

        """
    domain_list = []
    group_list = []
    group_id_list = []
    for _ in range(3):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        domain_list.append(domain)
        group = unit.new_group_ref(domain_id=domain['id'])
        group = PROVIDERS.identity_api.create_group(group)
        group_list.append(group)
        group_id_list.append(group['id'])
    role1 = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role1['id'], role1)
    PROVIDERS.assignment_api.create_grant(group_id=group_list[0]['id'], domain_id=domain_list[0]['id'], role_id=role1['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group_list[1]['id'], domain_id=domain_list[1]['id'], role_id=role1['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group_list[2]['id'], domain_id=domain_list[2]['id'], role_id=role1['id'], inherited_to_projects=True)
    domain_refs = PROVIDERS.assignment_api.list_domains_for_groups(group_id_list)
    self.assertThat(domain_refs, matchers.HasLength(2))
    self.assertIn(domain_list[0], domain_refs)
    self.assertIn(domain_list[1], domain_refs)