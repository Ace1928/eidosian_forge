from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import clusters as api_cl
from saharaclient.api import images as api_img
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import clusters as osc_cl
from saharaclient.tests.unit.osc.v1 import test_clusters as tc_v1
def test_cluster_create_with_count(self):
    clusters_mock = mock.Mock()
    clusters_mock.to_dict.return_value = {'clusters': [{'cluster': {'id': 'cluster1_id'}}, {'cluster': {'id': 'cluster2_id'}}]}
    self.cl_mock.create.return_value = clusters_mock
    arglist = ['--name', 'fake', '--cluster-template', 'template', '--image', 'ubuntu', '--count', '2']
    verifylist = [('name', 'fake'), ('cluster_template', 'template'), ('image', 'ubuntu'), ('count', 2)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.cl_mock.create.assert_called_once_with(cluster_template_id='ct_id', count=2, default_image_id='img_id', description=None, plugin_version='0.1', is_protected=False, is_public=False, is_transient=False, name='fake', net_id=None, plugin_name='fake', user_keypair_id=None)
    expected_columns = ('fake',)
    self.assertEqual(expected_columns, columns)
    expected_data = ('cluster_id',)
    self.assertEqual(expected_data, data)