from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_message
def test_message_delete_multiple_messages(self):
    self.volume_client.api_version = api_versions.APIVersion('3.3')
    arglist = [self.fake_messages[0].id, self.fake_messages[1].id]
    verifylist = [('message_ids', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for m in self.fake_messages:
        calls.append(call(m.id))
    self.volume_messages_mock.delete.assert_has_calls(calls)
    self.assertIsNone(result)