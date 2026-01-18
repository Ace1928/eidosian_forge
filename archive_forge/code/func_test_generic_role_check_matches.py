from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_generic_role_check_matches(self):
    check = _checks.GenericCheck('token.roles.name', 'role1')
    credentials = token_fixture.PROJECT_SCOPED_TOKEN_FIXTURE
    self.assertTrue(check({}, credentials, self.enforcer))