from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_set_property(self):
    self.volume_client.api_version = api_versions.APIVersion('3.43')
    arglist = ['--property', 'foo=bar', self.backup.id]
    verifylist = [('properties', {'foo': 'bar'}), ('backup', self.backup.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'metadata': {'wow': 'cool', 'foo': 'bar'}}
    self.backups_mock.update.assert_called_once_with(self.backup.id, **kwargs)
    self.assertIsNone(result)