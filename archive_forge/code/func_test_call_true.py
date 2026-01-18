from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_call_true(self):
    rule = _checks.TrueCheck()
    check = _checks.NotCheck(rule)
    self.assertFalse(check('target', 'cred', None))