from unittest import mock
from openstackclient.network.v2 import (
from openstackclient.tests.unit.network.v2 import fakes
def test_network_service_provider_list(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.service_providers.assert_called_with()
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))