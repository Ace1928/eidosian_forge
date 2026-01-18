from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.create')
def test_cluster_template_create_success_with_registry_enabled(self, mock_create):
    self._test_arg_success('cluster-template-create --name test --network-driver test_driver --keypair-id test_keypair --external-network-id test_net --image-id test_image --coe swarm --registry-enabled')
    expected_args = self._get_expected_args(name='test', image_id='test_image', keypair_id='test_keypair', coe='swarm', external_network_id='test_net', network_driver='test_driver', registry_enabled=True)
    mock_create.assert_called_with(**expected_args)