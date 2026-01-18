from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
def test_backup_list_by_project(self):
    parsed_args = self.check_parser(self.cmd, ['--project-id', 'fake_id'], [])
    self.cmd.take_action(parsed_args)
    params = {'datastore': None, 'limit': None, 'marker': None, 'instance_id': None, 'all_projects': False, 'project_id': 'fake_id'}
    self.backup_client.list.assert_called_once_with(**params)