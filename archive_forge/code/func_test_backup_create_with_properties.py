from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_create_with_properties(self):
    self._set_mock_microversion('3.43')
    arglist = ['--property', 'foo=bar', '--property', 'wow=much-cool', self.new_backup.volume_id]
    verifylist = [('properties', {'foo': 'bar', 'wow': 'much-cool'}), ('volume', self.new_backup.volume_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_sdk_client.create_backup.assert_called_with(volume_id=self.new_backup.volume_id, container=None, name=None, description=None, force=False, incremental=False, metadata={'foo': 'bar', 'wow': 'much-cool'})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)