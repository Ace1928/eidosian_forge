from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
def test_create_from_data_location(self):
    name = self.random_name('backup')
    ds_version = self.random_uuid()
    args = [name, '--restore-from', 'fake-remote-location', '--restore-datastore-version', ds_version, '--restore-size', '3']
    parsed_args = self.check_parser(self.cmd, args, [])
    self.cmd.take_action(parsed_args)
    self.backup_client.create.assert_called_with(name, None, restore_from='fake-remote-location', restore_ds_version=ds_version, restore_size=3)