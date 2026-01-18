from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
def test_A_and_B_or_C_with_not_1(self):
    result = _parser._parse_text_rule('not @ and ! or @')
    self.assertEqual('((not @ and !) or @)', str(result))