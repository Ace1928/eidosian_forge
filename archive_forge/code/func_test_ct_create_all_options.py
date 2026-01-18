from osc_lib.tests import utils as osc_utils
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import cluster_templates as osc_ct
from saharaclient.tests.unit.osc.v1 import test_cluster_templates as tct_v1
def test_ct_create_all_options(self):
    arglist = ['--name', 'template', '--node-groups', 'fakeng:2', '--anti-affinity', 'datanode', '--description', 'descr', '--autoconfig', '--public', '--protected', '--domain-name', 'domain.org.']
    verifylist = [('name', 'template'), ('node_groups', ['fakeng:2']), ('description', 'descr'), ('autoconfig', True), ('public', True), ('protected', True), ('domain_name', 'domain.org.')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.ct_mock.create.assert_called_once_with(description='descr', plugin_version='0.1', is_protected=True, is_public=True, name='template', node_groups=[{'count': 2, 'name': 'fakeng', 'node_group_template_id': 'd29631fc-0fad-434b-80aa-7a3e9526f57c'}], plugin_name='fake', use_autoconfig=True, shares=None, cluster_configs=None, domain_name='domain.org.')
    expected_columns = ('Anti affinity', 'Description', 'Domain name', 'Id', 'Is default', 'Is protected', 'Is public', 'Name', 'Node groups', 'Plugin name', 'Plugin version', 'Use autoconfig')
    self.assertEqual(expected_columns, columns)
    expected_data = ('', 'Cluster template for tests', 'domain.org.', '0647061f-ab98-4c89-84e0-30738ea55750', False, False, False, 'template', 'fakeng:2', 'fake', '0.1', True)
    self.assertEqual(expected_data, data)