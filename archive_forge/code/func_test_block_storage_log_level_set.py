from cinderclient import api_versions
import ddt
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_log_level as service
def test_block_storage_log_level_set(self):
    self.volume_client.api_version = api_versions.APIVersion('3.32')
    arglist = ['ERROR', '--host', self.service_log.host, '--service', self.service_log.binary, '--log-prefix', self.service_log.prefix]
    verifylist = [('level', 'ERROR'), ('host', self.service_log.host), ('service', self.service_log.binary), ('log_prefix', self.service_log.prefix)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.service_mock.set_log_levels.assert_called_with(level='ERROR', server=self.service_log.host, binary=self.service_log.binary, prefix=self.service_log.prefix)