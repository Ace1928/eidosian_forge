from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_flavor_by_id(self):
    self.use_compute_discovery()
    flavor_uri = '{endpoint}/flavors/1'.format(endpoint=fakes.COMPUTE_ENDPOINT)
    flavor_json = {'flavor': fakes.make_fake_flavor('1', 'vanilla')}
    self.register_uris([dict(method='GET', uri=flavor_uri, json=flavor_json)])
    flavor1 = self.cloud.get_flavor_by_id('1')
    self.assertEqual('1', flavor1['id'])
    self.assertEqual({}, flavor1.extra_specs)
    flavor2 = self.cloud.get_flavor_by_id('1')
    self.assertEqual('1', flavor2['id'])
    self.assertEqual({}, flavor2.extra_specs)