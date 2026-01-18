from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import ndp_proxy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_ndp_proxy_list(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.ndp_proxies.assert_called_once_with()
    self.assertEqual(self.columns, columns)
    list_data = list(data)
    self.assertEqual(len(self.data), len(list_data))
    for index in range(len(list_data)):
        self.assertEqual(self.data[index], list_data[index])