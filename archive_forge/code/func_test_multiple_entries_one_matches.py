from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_multiple_entries_one_matches(self):
    check = _checks.GenericCheck('token.catalog.endpoints.id', token_fixture.REGION_ONE_PUBLIC_KEYSTONE_ENDPOINT_ID)
    credentials = token_fixture.PROJECT_SCOPED_TOKEN_FIXTURE
    self.assertTrue(check({}, credentials, self.enforcer))