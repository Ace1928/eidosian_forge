from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_call_all_false(self):
    rules = [_BoolCheck(False), _BoolCheck(False)]
    check = _checks.OrCheck(rules)
    self.assertFalse(check('target', 'cred', None))
    self.assertTrue(rules[0].called)
    self.assertTrue(rules[1].called)