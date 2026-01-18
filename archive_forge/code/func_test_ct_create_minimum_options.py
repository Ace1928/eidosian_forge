from osc_lib.tests import utils as osc_utils
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import cluster_templates as osc_ct
from saharaclient.tests.unit.osc.v1 import test_cluster_templates as tct_v1
def test_ct_create_minimum_options(self):
    arglist = ['--name', 'template', '--node-groups', 'fakeng:2']
    verifylist = [('name', 'template'), ('node_groups', ['fakeng:2'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.ct_mock.create.assert_called_once_with(description=None, plugin_version='0.1', is_protected=False, is_public=False, name='template', node_groups=[{'count': 2, 'name': 'fakeng', 'node_group_template_id': 'd29631fc-0fad-434b-80aa-7a3e9526f57c'}], plugin_name='fake', use_autoconfig=False, shares=None, cluster_configs=None, domain_name=None)