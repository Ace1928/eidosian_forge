from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.update')
def test_cluster_template_update_label(self, mock_update):
    self._test_arg_success('cluster-template-update test replace labels=key1=val1')
    patch = [{'op': 'replace', 'path': '/labels', 'value': "{'key1': 'val1'}"}]
    mock_update.assert_called_once_with('test', patch)