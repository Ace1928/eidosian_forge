from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_parser, '_parse_check', base.FakeCheck)
def test_multi_multi(self):
    result = _parser._parse_list_rule([['rule1', 'rule2'], ['rule3', 'rule4']])
    self.assertIsInstance(result, _checks.OrCheck)
    self.assertEqual(2, len(result.rules))
    for i, values in enumerate([['rule1', 'rule2'], ['rule3', 'rule4']]):
        self.assertIsInstance(result.rules[i], _checks.AndCheck)
        self.assertEqual(2, len(result.rules[i].rules))
        for j, value in enumerate(values):
            self.assertIsInstance(result.rules[i].rules[j], base.FakeCheck)
            self.assertEqual(value, result.rules[i].rules[j].result)
    self.assertEqual('((rule1 and rule2) or (rule3 and rule4))', str(result))