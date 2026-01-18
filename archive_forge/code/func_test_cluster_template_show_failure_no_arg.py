from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.get')
def test_cluster_template_show_failure_no_arg(self, mock_show):
    self._test_arg_failure('cluster-template-show', self._few_argument_error)
    mock_show.assert_not_called()