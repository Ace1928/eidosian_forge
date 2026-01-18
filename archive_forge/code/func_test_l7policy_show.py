import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_l7policy_attrs')
def test_l7policy_show(self, mock_attrs):
    mock_attrs.return_value = {'l7policy_id': self._l7po.id}
    arglist = [self._l7po.id]
    verifylist = [('l7policy', self._l7po.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7policy_show.assert_called_with(l7policy_id=self._l7po.id)