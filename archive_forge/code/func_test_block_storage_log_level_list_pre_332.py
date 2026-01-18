from cinderclient import api_versions
import ddt
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_log_level as service
def test_block_storage_log_level_list_pre_332(self):
    arglist = ['--host', self.service_log.host, '--service', 'cinder-api', '--log-prefix', 'cinder_test.api.common']
    verifylist = [('host', self.service_log.host), ('service', 'cinder-api'), ('log_prefix', 'cinder_test.api.common')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.32 or greater is required', str(exc))