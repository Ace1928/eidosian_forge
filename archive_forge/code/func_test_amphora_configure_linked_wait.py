import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import amphora
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_status')
def test_amphora_configure_linked_wait(self, mock_wait):
    arglist = [self._amp.id, '--wait']
    verify_list = [('amphora_id', self._amp.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verify_list)
    self.cmd.take_action(parsed_args)
    self.api_mock.amphora_configure.assert_called_with(amphora_id=self._amp.id)
    mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._amp.loadbalancer_id, sleep_time=mock.ANY, status_field='provisioning_status')