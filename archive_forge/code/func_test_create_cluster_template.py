import testtools
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack import exceptions
from openstack.tests.unit import base
def test_create_cluster_template(self):
    json_response = cluster_template_obj.copy()
    kwargs = dict(name=cluster_template_obj['name'], image_id=cluster_template_obj['image_id'], keypair_id=cluster_template_obj['keypair_id'], coe=cluster_template_obj['coe'])
    self.register_uris([dict(method='POST', uri=self.get_mock_url(resource='clustertemplates'), json=json_response, validate=dict(json=kwargs))])
    response = self.cloud.create_cluster_template(**kwargs)
    self._compare_clustertemplates(json_response, response)
    self.assert_calls()