from unittest import mock
from openstackclient.compute.v2 import console
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils
def test_console_url_show_with_novnc(self):
    arglist = ['--novnc', 'foo_vm']
    verifylist = [('url_type', 'novnc'), ('server', 'foo_vm')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.create_console.assert_called_once_with(self._server.id, console_type='novnc')
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)