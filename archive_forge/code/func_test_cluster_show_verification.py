from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import clusters as api_cl
from saharaclient.api import images as api_img
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import clusters as osc_cl
from saharaclient.tests.unit.osc.v1 import test_clusters as tc_v1
def test_cluster_show_verification(self):
    arglist = ['fake', '--verification']
    verifylist = [('cluster', 'fake'), ('verification', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.cl_mock.find_unique.assert_called_once_with(name='fake')
    expected_columns = ('Anti affinity', 'Cluster template id', 'Description', 'Health check (some check)', 'Id', 'Image', 'Is protected', 'Is public', 'Name', 'Neutron management network', 'Node groups', 'Plugin name', 'Plugin version', 'Status', 'Use autoconfig', 'User keypair id', 'Verification status')
    self.assertEqual(expected_columns, columns)
    expected_data = ('', 'ct_id', 'Cluster template for tests', 'GREEN', 'cluster_id', 'img_id', False, False, 'fake', 'net_id', 'fakeng:2', 'fake', '0.1', 'Active', True, 'test', 'GREEN')
    self.assertEqual(expected_data, data)