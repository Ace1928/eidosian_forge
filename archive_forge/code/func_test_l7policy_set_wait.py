import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_status')
def test_l7policy_set_wait(self, mock_wait):
    arglist = [self._l7po.id, '--name', 'new_name', '--wait']
    verifylist = [('l7policy', self._l7po.id), ('name', 'new_name'), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7policy_set.assert_called_with(self._l7po.id, json={'l7policy': {'name': 'new_name'}})
    mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._l7po.id, sleep_time=mock.ANY, status_field='provisioning_status')