from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
def test_A_and_B_or_C_with_group_and_not_4(self):
    for expression in ['( @ ) and not ! or @', '@ and ( not ! ) or @', '@ and not ( ! ) or @', '@ and not ! or ( @ )', '( @ and not ! ) or @', '( @ and not ! or @ )']:
        result = _parser._parse_text_rule(expression)
        self.assertEqual('((@ and not !) or @)', str(result))