from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def create_assignments(self, assignment_pattern, test_data):
    """Create the assignments specified in the test plan."""
    test_data['initial_assignment_count'] = len(PROVIDERS.assignment_api.list_role_assignments())
    for assignment in assignment_pattern:
        args = {}
        for param in assignment:
            if param == 'inherited_to_projects':
                args[param] = assignment[param]
            else:
                key, value = self._convert_entity_shorthand(param, assignment, test_data)
                args[key] = value
        PROVIDERS.assignment_api.create_grant(**args)
    return test_data