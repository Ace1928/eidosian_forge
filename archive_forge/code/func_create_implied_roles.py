from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def create_implied_roles(self, implied_pattern, test_data):
    """Create the implied roles specified in the test plan."""
    for implied_spec in implied_pattern:
        prior_role = test_data['roles'][implied_spec['role']]['id']
        if isinstance(implied_spec['implied_roles'], list):
            for this_role in implied_spec['implied_roles']:
                implied_role = test_data['roles'][this_role]['id']
                PROVIDERS.role_api.create_implied_role(prior_role, implied_role)
        else:
            implied_role = test_data['roles'][implied_spec['implied_roles']]['id']
            PROVIDERS.role_api.create_implied_role(prior_role, implied_role)