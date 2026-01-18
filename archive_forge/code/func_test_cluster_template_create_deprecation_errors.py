from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.create')
def test_cluster_template_create_deprecation_errors(self, mock_create):
    required_args = 'cluster-template-create --coe test --external-network public --image test '
    self._test_arg_failure('cluster-template-create --coe test --external-network-id test --external-network test ', self._too_many_group_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure('cluster-template-create --coe test --image-id test --image test ', self._too_many_group_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure(required_args + '--flavor test --flavor-id test', self._too_many_group_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure(required_args + '--master-flavor test --master-flavor-id test', self._too_many_group_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure(required_args + '--keypair test --keypair-id test', self._too_many_group_arg_error)
    mock_create.assert_not_called()