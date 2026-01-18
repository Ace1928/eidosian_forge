from openstack.container_infrastructure_management.v1 import cluster
from openstack.tests.unit import base
def test_delete_coe_cluster(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clusters'), json=dict(clusters=[coe_cluster_obj])), dict(method='DELETE', uri=self.get_mock_url(resource='clusters', append=[coe_cluster_obj['uuid']]))])
    self.cloud.delete_coe_cluster(coe_cluster_obj['uuid'])
    self.assert_calls()