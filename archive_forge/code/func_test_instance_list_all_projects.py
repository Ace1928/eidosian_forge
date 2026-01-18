from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
def test_instance_list_all_projects(self):
    instance_id = self.random_uuid()
    name = self.random_name('test-list')
    tenant_id = self.random_uuid()
    server_id = self.random_uuid()
    insts = [{'id': instance_id, 'name': name, 'status': 'ACTIVE', 'operating_status': 'HEALTHY', 'addresses': [{'type': 'private', 'address': '10.0.0.13'}], 'volume': {'size': 2}, 'flavor': {'id': '02'}, 'region': 'regionOne', 'datastore': {'version': '5.6', 'type': 'mysql', 'version_number': '5.7.29'}, 'tenant_id': tenant_id, 'access': {'is_public': False, 'allowed_cidrs': []}, 'server_id': server_id, 'server': {'id': server_id}}]
    self.mgmt_client.list.return_value = common.Paginated([instances.Instance(mock.MagicMock(), inst) for inst in insts])
    parsed_args = self.check_parser(self.cmd, ['--all-projects'], [('all_projects', True)])
    columns, data = self.cmd.take_action(parsed_args)
    self.mgmt_client.list.assert_called_once_with(**self.defaults)
    self.assertEqual(database_instances.ListDatabaseInstances.admin_columns, columns)
    expected_instances = [(instance_id, name, 'mysql', '5.6', 'ACTIVE', 'HEALTHY', False, [{'type': 'private', 'address': '10.0.0.13'}], '02', 2, '', server_id, tenant_id)]
    self.assertEqual(expected_instances, data)