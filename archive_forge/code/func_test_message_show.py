from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import messages as osc_messages
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_message_show(self):
    arglist = [self.message.id]
    verifylist = [('message', self.message.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.messages_mock.get.assert_called_with(self.message.id)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)