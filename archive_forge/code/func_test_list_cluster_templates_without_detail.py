import testtools
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack import exceptions
from openstack.tests.unit import base
def test_list_cluster_templates_without_detail(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj]))])
    cluster_templates_list = self.cloud.list_cluster_templates()
    self._compare_clustertemplates(cluster_template_obj, cluster_templates_list[0])
    self.assert_calls()