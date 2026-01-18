from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_local_ip_list_name(self):
    arglist = ['--name', self.local_ips[0].name]
    verifylist = [('name', self.local_ips[0].name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.local_ips.assert_called_once_with(**{'name': self.local_ips[0].name})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))