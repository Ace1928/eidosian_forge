from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_flavor_string_and_int(self):
    self.use_compute_discovery()
    flavor_resource_uri = '{endpoint}/flavors/1/os-extra_specs'.format(endpoint=fakes.COMPUTE_ENDPOINT)
    flavor = fakes.make_fake_flavor('1', 'vanilla')
    flavor_json = {'extra_specs': {}}
    self.register_uris([dict(method='GET', uri='{endpoint}/flavors/1'.format(endpoint=fakes.COMPUTE_ENDPOINT), json=flavor), dict(method='GET', uri=flavor_resource_uri, json=flavor_json)])
    flavor1 = self.cloud.get_flavor('1')
    self.assertEqual('1', flavor1['id'])
    flavor2 = self.cloud.get_flavor(1)
    self.assertEqual('1', flavor2['id'])