from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_local_ips_delete_with_exception(self):
    arglist = [self._local_ips[0].name, 'unexist_local_ip']
    verifylist = [('local_ip', [self._local_ips[0].name, 'unexist_local_ip'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [self._local_ips[0], exceptions.CommandError]
    self.network_client.find_local_ip = mock.Mock(side_effect=find_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 local IPs failed to delete.', str(e))
    self.network_client.find_local_ip.assert_any_call(self._local_ips[0].name, ignore_missing=False)
    self.network_client.find_local_ip.assert_any_call('unexist_local_ip', ignore_missing=False)
    self.network_client.delete_local_ip.assert_called_once_with(self._local_ips[0])