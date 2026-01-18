import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_group_duplicate_conflict_gives_name(self):
    group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
    group = PROVIDERS.identity_api.create_group(group)
    try:
        PROVIDERS.identity_api.create_group(group)
    except exception.Conflict as e:
        self.assertIn('Duplicate entry found with name %s' % group['name'], repr(e))
    else:
        self.fail('Create duplicate group did not raise a conflict')