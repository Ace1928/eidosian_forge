from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import messages as osc_messages
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_message_delete_multiple(self):
    messages = manila_fakes.FakeMessage.create_messages(count=2)
    arglist = [messages[0].id, messages[1].id]
    verifylist = [('message', [messages[0].id, messages[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertEqual(self.messages_mock.delete.call_count, len(messages))
    self.assertIsNone(result)