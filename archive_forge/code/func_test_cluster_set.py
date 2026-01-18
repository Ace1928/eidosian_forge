from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_cluster
def test_cluster_set(self):
    self.volume_client.api_version = api_versions.APIVersion('3.7')
    arglist = ['--enable', self.cluster.name]
    verifylist = [('cluster', self.cluster.name), ('binary', 'cinder-volume'), ('disabled', False), ('disabled_reason', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, tuple(data))
    self.cluster_mock.update.assert_called_once_with(self.cluster.name, 'cinder-volume', disabled=False, disabled_reason=None)