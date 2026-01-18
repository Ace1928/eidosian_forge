from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_flavor_by_ram_not_found(self):
    self.use_compute_discovery()
    self.register_uris([dict(method='GET', uri='{endpoint}/flavors/detail?is_public=None'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'flavors': []})])
    self.assertRaises(exceptions.SDKException, self.cloud.get_flavor_by_ram, ram=100)