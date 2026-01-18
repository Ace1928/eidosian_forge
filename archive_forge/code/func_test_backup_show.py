from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_show(self):
    arglist = [self.backup.id]
    verifylist = [('backup', self.backup.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_sdk_client.get_backup.assert_called_with(self.backup.id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)