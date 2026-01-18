from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.create')
def test_cluster_template_create_success_only_positional_name(self, mock_create):
    self._test_arg_success('cluster-template-create test --labels key1=val1,key2=val2 --keypair-id test_keypair --external-network-id test_net --image-id test_image --coe swarm --server-type vm')
    expected_args = self._get_expected_args(name='test', image_id='test_image', keypair_id='test_keypair', coe='swarm', external_network_id='test_net', server_type='vm', labels={'key1': 'val1', 'key2': 'val2'})
    mock_create.assert_called_with(**expected_args)