from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_checks, 'NotCheck', lambda x: 'not %s' % x)
def test_make_not_expr(self):
    state = _parser.ParseState()
    result = state._make_not_expr('not', 'check')
    self.assertEqual([('check', 'not check')], result)