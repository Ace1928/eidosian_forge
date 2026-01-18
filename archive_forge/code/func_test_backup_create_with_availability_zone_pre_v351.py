from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_create_with_availability_zone_pre_v351(self):
    self._set_mock_microversion('3.50')
    arglist = ['--availability-zone', 'my-az', self.new_backup.volume_id]
    verifylist = [('availability_zone', 'my-az'), ('volume', self.new_backup.volume_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.51 or greater', str(exc))