from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import node_group_templates as osc_ngt
from saharaclient.tests.unit.osc.v1 import fakes
def test_ngt_create_minimum_options(self):
    arglist = ['--name', 'template', '--plugin', 'fake', '--plugin-version', '0.1', '--processes', 'namenode', 'tasktracker', '--flavor', 'flavor_id']
    verifylist = [('name', 'template'), ('plugin', 'fake'), ('plugin_version', '0.1'), ('flavor', 'flavor_id'), ('processes', ['namenode', 'tasktracker'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.ngt_mock.create.assert_called_once_with(auto_security_group=False, availability_zone=None, description=None, flavor_id='flavor_id', floating_ip_pool=None, plugin_version='0.1', is_protected=False, is_proxy_gateway=False, is_public=False, name='template', node_processes=['namenode', 'tasktracker'], plugin_name='fake', security_groups=None, use_autoconfig=False, volume_local_to_instance=False, volume_type=None, volumes_availability_zone=None, volumes_per_node=None, volumes_size=None, shares=None, node_configs=None, volume_mount_prefix=None, boot_from_volume=False, boot_volume_type=None, boot_volume_availability_zone=None, boot_volume_local_to_instance=False)