from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_cluster
def test_cluster_set_pre_v37(self):
    self.volume_client.api_version = api_versions.APIVersion('3.6')
    arglist = ['--enable', self.cluster.name]
    verifylist = [('cluster', self.cluster.name), ('binary', 'cinder-volume'), ('disabled', False), ('disabled_reason', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.7 or greater is required', str(exc))