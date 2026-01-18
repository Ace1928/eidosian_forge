import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_status')
def test_listener_set_wait(self, mock_wait):
    arglist = [self._listener.id, '--name', 'new_name', '--wait']
    verifylist = [('listener', self._listener.id), ('name', 'new_name'), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.listener_set.assert_called_with(self._listener.id, json={'listener': {'name': 'new_name'}})
    mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._listener.id, sleep_time=mock.ANY, status_field='provisioning_status')