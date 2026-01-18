from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_multi_subnet_pools_delete_with_exception(self):
    arglist = [self._subnet_pools[0].name, 'unexist_subnet_pool']
    verifylist = [('subnet_pool', [self._subnet_pools[0].name, 'unexist_subnet_pool'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [self._subnet_pools[0], exceptions.CommandError]
    self.network_client.find_subnet_pool = mock.Mock(side_effect=find_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 subnet pools failed to delete.', str(e))
    self.network_client.find_subnet_pool.assert_any_call(self._subnet_pools[0].name, ignore_missing=False)
    self.network_client.find_subnet_pool.assert_any_call('unexist_subnet_pool', ignore_missing=False)
    self.network_client.delete_subnet_pool.assert_called_once_with(self._subnet_pools[0])