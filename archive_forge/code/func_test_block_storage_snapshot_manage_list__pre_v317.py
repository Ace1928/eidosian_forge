from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as v2_volume_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_manage
def test_block_storage_snapshot_manage_list__pre_v317(self):
    self.volume_client.api_version = api_versions.APIVersion('3.16')
    arglist = ['--cluster', 'fake_cluster']
    verifylist = [('cluster', 'fake_cluster')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.17 or greater is required', str(exc))
    self.assertIn('--cluster', str(exc))