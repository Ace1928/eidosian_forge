from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
def test_get_cluster_id(self):
    mock_cluster = mock.Mock(id='fake_cluster_id')
    mock_get = self.patchobject(self.client, 'get_cluster', return_value=mock_cluster)
    ret = self.plugin.get_cluster_id('fake_cluster')
    self.assertEqual('fake_cluster_id', ret)
    mock_get.assert_called_once_with('fake_cluster')