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
def test_get_cluster_policy(self):
    fake_policy = cluster_policy.ClusterPolicy.new(id='FAKE_POLICY')
    fake_cluster = cluster.Cluster.new(id='FAKE_CLUSTER')
    self._verify('openstack.proxy.Proxy._get', self.proxy.get_cluster_policy, method_args=[fake_policy, 'FAKE_CLUSTER'], expected_args=[cluster_policy.ClusterPolicy, fake_policy], expected_kwargs={'cluster_id': 'FAKE_CLUSTER'}, expected_result=fake_policy)
    self._verify('openstack.proxy.Proxy._get', self.proxy.get_cluster_policy, method_args=['FAKE_POLICY', 'FAKE_CLUSTER'], expected_args=[cluster_policy.ClusterPolicy, 'FAKE_POLICY'], expected_kwargs={'cluster_id': 'FAKE_CLUSTER'})
    self._verify('openstack.proxy.Proxy._get', self.proxy.get_cluster_policy, method_args=['FAKE_POLICY', fake_cluster], expected_args=[cluster_policy.ClusterPolicy, 'FAKE_POLICY'], expected_kwargs={'cluster_id': fake_cluster})