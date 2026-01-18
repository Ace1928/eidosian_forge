from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_role_assignments_user_not_found(self):

    def _user_not_found(value):
        raise exception.UserNotFound(user_id=value)
    with mock.patch.object(PROVIDERS.identity_api, 'get_user', _user_not_found):
        assignment_list = PROVIDERS.assignment_api.list_role_assignments(include_names=True)
    self.assertNotEqual([], assignment_list)
    for assignment in assignment_list:
        if 'user_name' in assignment:
            self.assertEqual('', assignment['user_name'])
            self.assertEqual('', assignment['user_domain_id'])
            self.assertEqual('', assignment['user_domain_name'])