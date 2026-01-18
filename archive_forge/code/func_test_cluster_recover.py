from unittest import mock
from openstack.clustering.v1 import _proxy
from openstack.clustering.v1 import action
from openstack.clustering.v1 import build_info
from openstack.clustering.v1 import cluster
from openstack.clustering.v1 import cluster_attr
from openstack.clustering.v1 import cluster_policy
from openstack.clustering.v1 import event
from openstack.clustering.v1 import node
from openstack.clustering.v1 import policy
from openstack.clustering.v1 import policy_type
from openstack.clustering.v1 import profile
from openstack.clustering.v1 import profile_type
from openstack.clustering.v1 import receiver
from openstack.clustering.v1 import service
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
@mock.patch.object(proxy_base.Proxy, '_get_resource')
def test_cluster_recover(self, mock_get):
    mock_cluster = cluster.Cluster.new(id='FAKE_CLUSTER')
    mock_get.return_value = mock_cluster
    self._verify('openstack.clustering.v1.cluster.Cluster.recover', self.proxy.recover_cluster, method_args=['FAKE_CLUSTER'], expected_args=[self.proxy])
    mock_get.assert_called_once_with(cluster.Cluster, 'FAKE_CLUSTER')