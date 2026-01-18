import testtools
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack import exceptions
from openstack.tests.unit import base
def test_delete_cluster_template(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj])), dict(method='DELETE', uri=self.get_mock_url(resource='clustertemplates/fake-uuid'))])
    self.cloud.delete_cluster_template('fake-uuid')
    self.assert_calls()