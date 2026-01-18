from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_missing_credentials_dictionary_lookup(self):
    credentials = {'a': 'APPLES', 'o': {'t': 'ORANGES'}}
    check = _checks.GenericCheck('o.t', 'ORANGES')
    self.assertTrue(check({}, credentials, self.enforcer))
    check = _checks.GenericCheck('o.v', 'ORANGES')
    self.assertFalse(check({}, credentials, self.enforcer))
    check = _checks.GenericCheck('q.v', 'APPLES')
    self.assertFalse(check({}, credentials, self.enforcer))