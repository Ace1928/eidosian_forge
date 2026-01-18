from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_policy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_qos_policies_delete_with_exception(self):
    arglist = [self._qos_policies[0].name, 'unexist_qos_policy']
    verifylist = [('policy', [self._qos_policies[0].name, 'unexist_qos_policy'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [self._qos_policies[0], exceptions.CommandError]
    self.network_client.find_qos_policy = mock.MagicMock(side_effect=find_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 QoS policies failed to delete.', str(e))
    self.network_client.find_qos_policy.assert_any_call(self._qos_policies[0].name, ignore_missing=False)
    self.network_client.find_qos_policy.assert_any_call('unexist_qos_policy', ignore_missing=False)
    self.network_client.delete_qos_policy.assert_called_once_with(self._qos_policies[0])