from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
def test_A_and_B_and_C_and_D(self):
    result = _parser._parse_text_rule('@ and ! and @ and !')
    self.assertEqual('(@ and ! and @ and !)', str(result))