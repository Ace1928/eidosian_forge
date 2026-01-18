from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
@mock.patch.object(utils, 'find_resource')
def test_instance_create_without_allowed_cidrs(self, mock_find):
    resp = {'id': 'a1fea1cf-18ad-48ab-bdfd-fce99a4b834e', 'name': 'test-mysql', 'status': 'BUILD', 'flavor': {'id': 'a48ea749-7ee3-4003-8aae-eb4e79773e2d'}, 'datastore': {'type': 'mysql', 'version': '5.7.29', 'version_number': '5.7.29'}, 'region': 'RegionOne', 'access': {'is_public': True}, 'volume': {'size': 1}, 'created': '2020-08-12T09:41:47', 'updated': '2020-08-12T09:41:47', 'service_status_updated': '2020-08-12T09:41:47'}
    self.instance_client.create.return_value = instances.Instance(mock.MagicMock(), resp)
    args = ['test-mysql', '--flavor', 'a48ea749-7ee3-4003-8aae-eb4e79773e2d', '--size', '1', '--datastore', 'mysql', '--datastore-version', '5.7.29', '--nic', 'net-id=net1', '--is-public']
    verifylist = [('name', 'test-mysql'), ('flavor', 'a48ea749-7ee3-4003-8aae-eb4e79773e2d'), ('size', 1), ('datastore', 'mysql'), ('datastore_version', '5.7.29'), ('nics', 'net-id=net1'), ('is_public', True), ('allowed_cidrs', None)]
    parsed_args = self.check_parser(self.cmd, args, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ('allowed_cidrs', 'created', 'datastore', 'datastore_version', 'datastore_version_number', 'flavor', 'id', 'name', 'public', 'region', 'service_status_updated', 'status', 'updated', 'volume')
    expected_values = ([], '2020-08-12T09:41:47', 'mysql', '5.7.29', '5.7.29', 'a48ea749-7ee3-4003-8aae-eb4e79773e2d', 'a1fea1cf-18ad-48ab-bdfd-fce99a4b834e', 'test-mysql', True, 'RegionOne', '2020-08-12T09:41:47', 'BUILD', '2020-08-12T09:41:47', 1)
    self.assertEqual(expected_columns, columns)
    self.assertEqual(expected_values, data)