from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_delete_multi_with_exception(self, net_mock):
    net_mock.return_value = mock.Mock(return_value=None)
    net_mock.side_effect = [mock.Mock(return_value=None), exceptions.CommandError]
    arglist = [self._networks[0]['id'], 'xxxx-yyyy-zzzz', self._networks[1]['id']]
    verifylist = [('network', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('2 of 3 networks failed to delete.', str(e))
    net_mock.assert_any_call(self._networks[0]['id'])
    net_mock.assert_any_call(self._networks[1]['id'])
    net_mock.assert_any_call('xxxx-yyyy-zzzz')