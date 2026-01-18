from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_flavor_by_ram_and_include(self):
    self.use_compute_discovery()
    uris_to_mock = [dict(method='GET', uri='{endpoint}/flavors/detail?is_public=None'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'flavors': fakes.FAKE_FLAVOR_LIST})]
    uris_to_mock.extend([dict(method='GET', uri='{endpoint}/flavors/{id}/os-extra_specs'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=flavor['id']), json={'extra_specs': {}}) for flavor in fakes.FAKE_FLAVOR_LIST])
    self.register_uris(uris_to_mock)
    flavor = self.cloud.get_flavor_by_ram(ram=150, include='strawberry')
    self.assertEqual(fakes.STRAWBERRY_FLAVOR_ID, flavor['id'])