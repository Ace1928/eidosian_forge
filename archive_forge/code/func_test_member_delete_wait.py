import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_delete')
@mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
def test_member_delete_wait(self, mock_attrs, mock_wait):
    mock_attrs.return_value = {'pool_id': self._mem.pool_id, 'member_id': self._mem.id}
    arglist = [self._mem.pool_id, self._mem.id, '--wait']
    verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.member_delete.assert_called_with(pool_id=self._mem.pool_id, member_id=self._mem.id)
    mock_wait.assert_called_once_with(manager=mock.ANY, res_id=self._mem.id, sleep_time=mock.ANY, status_field='provisioning_status')