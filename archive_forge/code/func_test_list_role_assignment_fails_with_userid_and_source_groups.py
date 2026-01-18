from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_role_assignment_fails_with_userid_and_source_groups(self):
    """Show we trap this unsupported internal combination of params."""
    group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
    group = PROVIDERS.identity_api.create_group(group)
    self.assertRaises(exception.UnexpectedError, PROVIDERS.assignment_api.list_role_assignments, effective=True, user_id=self.user_foo['id'], source_from_group_ids=[group['id']])