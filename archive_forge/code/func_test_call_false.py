from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_call_false(self):
    rule = _checks.FalseCheck()
    check = _checks.NotCheck(rule)
    self.assertTrue(check('target', 'cred', None))