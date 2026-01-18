from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_set_property_pre_v343(self):
    self.volume_client.api_version = api_versions.APIVersion('3.42')
    arglist = ['--property', 'foo=bar', self.backup.id]
    verifylist = [('properties', {'foo': 'bar'}), ('backup', self.backup.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.43 or greater', str(exc))