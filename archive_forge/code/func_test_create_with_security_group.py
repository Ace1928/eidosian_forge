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
def test_create_with_security_group(self):
    secgroup = network_fakes.FakeSecurityGroup.create_one_security_group()
    self.network_client.find_security_group = mock.Mock(return_value=secgroup)
    arglist = ['--network', self._port.network_id, '--security-group', secgroup.id, 'test-port']
    verifylist = [('network', self._port.network_id), ('enable', True), ('security_group', [secgroup.id]), ('name', 'test-port')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'security_group_ids': [secgroup.id], 'name': 'test-port'})
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)