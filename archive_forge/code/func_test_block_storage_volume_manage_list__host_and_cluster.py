from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as v2_volume_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_manage
def test_block_storage_volume_manage_list__host_and_cluster(self):
    self.volume_client.api_version = api_versions.APIVersion('3.17')
    arglist = ['fake_host', '--cluster', 'fake_cluster']
    verifylist = [('host', 'fake_host'), ('cluster', 'fake_cluster')]
    exc = self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
    self.assertIn('argument --cluster: not allowed with argument <host>', str(exc))