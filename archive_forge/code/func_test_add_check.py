from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_add_check(self):
    check = _checks.OrCheck(['rule1', 'rule2'])
    check.add_check('rule3')
    self.assertEqual(['rule1', 'rule2', 'rule3'], check.rules)