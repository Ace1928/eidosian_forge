from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_port_with_dns_name(self):
    arglist = ['--network', self._port.network_id, '--dns-name', '8.8.8.8', 'test-port']
    verifylist = [('network', self._port.network_id), ('enable', True), ('dns_name', '8.8.8.8'), ('name', 'test-port')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'dns_name': '8.8.8.8', 'name': 'test-port'})
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)