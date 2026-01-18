from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_checks, 'AndCheck', lambda x: x)
def test_make_and_expr(self):
    state = _parser.ParseState()
    result = state._make_and_expr('check1', 'and', 'check2')
    self.assertEqual([('and_expr', ['check1', 'check2'])], result)