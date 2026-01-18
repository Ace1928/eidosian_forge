from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_transfer_request
def test_delete_multiple_transfers(self):
    arglist = []
    for v in self.volume_transfers:
        arglist.append(v.id)
    verifylist = [('transfer_request', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for v in self.volume_transfers:
        calls.append(call(v.id))
    self.transfer_mock.delete.assert_has_calls(calls)
    self.assertIsNone(result)