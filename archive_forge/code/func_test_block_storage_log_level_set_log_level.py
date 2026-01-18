from cinderclient import api_versions
import ddt
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_log_level as service
@ddt.data('WARNING', 'info', 'Error', 'debuG', 'fake-log-level')
def test_block_storage_log_level_set_log_level(self, log_level):
    self.volume_client.api_version = api_versions.APIVersion('3.32')
    arglist = [log_level, '--host', self.service_log.host, '--service', 'cinder-api', '--log-prefix', 'cinder.api.common']
    verifylist = [('level', log_level.upper()), ('host', self.service_log.host), ('service', 'cinder-api'), ('log_prefix', 'cinder.api.common')]
    if log_level == 'fake-log-level':
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
    else:
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.service_mock.set_log_levels.assert_called_with(level=log_level.upper(), server=self.service_log.host, binary=self.service_log.binary, prefix=self.service_log.prefix)