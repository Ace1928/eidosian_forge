from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_address_groups_delete_with_exception(self):
    arglist = [self._address_groups[0].name, 'unexist_address_group']
    verifylist = [('address_group', [self._address_groups[0].name, 'unexist_address_group'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [self._address_groups[0], exceptions.CommandError]
    self.network_client.find_address_group = mock.Mock(side_effect=find_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 address groups failed to delete.', str(e))
    self.network_client.find_address_group.assert_any_call(self._address_groups[0].name, ignore_missing=False)
    self.network_client.find_address_group.assert_any_call('unexist_address_group', ignore_missing=False)
    self.network_client.delete_address_group.assert_called_once_with(self._address_groups[0])