from unittest import mock
from osprofiler.drivers.elasticsearch_driver import ElasticsearchDriver
from osprofiler.tests import test
def test_init_and_notify(self):
    self.elasticsearch.client = mock.MagicMock()
    self.elasticsearch.client.reset_mock()
    project = 'project'
    service = 'service'
    host = 'host'
    info = {'a': 10, 'project': project, 'service': service, 'host': host}
    self.elasticsearch.notify(info)
    self.elasticsearch.client.index.assert_called_once_with(index='osprofiler-notifications', doc_type='notification', body=info)