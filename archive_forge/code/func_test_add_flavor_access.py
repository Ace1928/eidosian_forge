from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_add_flavor_access(self):
    self.register_uris([dict(method='POST', uri='{endpoint}/flavors/{id}/action'.format(endpoint=fakes.COMPUTE_ENDPOINT, id='flavor_id'), json={'flavor_access': [{'flavor_id': 'flavor_id', 'tenant_id': 'tenant_id'}]}, validate=dict(json={'addTenantAccess': {'tenant': 'tenant_id'}}))])
    self.cloud.add_flavor_access('flavor_id', 'tenant_id')
    self.assert_calls()