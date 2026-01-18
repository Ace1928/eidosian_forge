import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_list_groups(self):
    group1 = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
    group2 = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
    group1 = PROVIDERS.identity_api.create_group(group1)
    group2 = PROVIDERS.identity_api.create_group(group2)
    groups = PROVIDERS.identity_api.list_groups(domain_scope=self._set_domain_scope(CONF.identity.default_domain_id))
    self.assertEqual(2, len(groups))
    group_ids = []
    for group in groups:
        group_ids.append(group.get('id'))
    self.assertIn(group1['id'], group_ids)
    self.assertIn(group2['id'], group_ids)