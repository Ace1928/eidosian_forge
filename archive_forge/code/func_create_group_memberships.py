from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def create_group_memberships(self, group_pattern, test_data):
    """Create the group memberships specified in the test plan."""
    for group_spec in group_pattern:
        group_value = test_data['groups'][group_spec['group']]['id']
        for user_index in group_spec['users']:
            user_value = test_data['users'][user_index]['id']
            PROVIDERS.identity_api.add_user_to_group(user_value, group_value)
    return test_data