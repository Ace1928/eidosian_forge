from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
@mock.patch('troveclient.utils.get_resource_id')
def test_backup_list_by_instance_name(self, get_resource_id_mock):
    get_resource_id_mock.return_value = 'fake_uuid'
    parsed_args = self.check_parser(self.cmd, ['--instance', 'fake_name'], [])
    self.cmd.take_action(parsed_args)
    params = {'datastore': None, 'limit': None, 'marker': None, 'instance_id': 'fake_uuid', 'all_projects': False, 'project_id': None}
    self.backup_client.list.assert_called_once_with(**params)
    get_resource_id_mock.assert_called_once_with(self.instance_client, 'fake_name')