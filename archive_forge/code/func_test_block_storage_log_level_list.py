from cinderclient import api_versions
import ddt
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_log_level as service
def test_block_storage_log_level_list(self):
    self.volume_client.api_version = api_versions.APIVersion('3.32')
    arglist = ['--host', self.service_log.host, '--service', self.service_log.binary, '--log-prefix', self.service_log.prefix]
    verifylist = [('host', self.service_log.host), ('service', self.service_log.binary), ('log_prefix', self.service_log.prefix)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['Binary', 'Host', 'Prefix', 'Level']
    self.assertEqual(expected_columns, columns)
    datalist = ((self.service_log.binary, self.service_log.host, self.service_log.prefix, self.service_log.level),)
    self.assertEqual(datalist, tuple(data))
    self.service_mock.get_log_levels.assert_called_with(server=self.service_log.host, binary=self.service_log.binary, prefix=self.service_log.prefix)