from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_flavors(self):
    self.use_compute_discovery()
    uris_to_mock = [dict(method='GET', uri='{endpoint}/flavors/detail?is_public=None'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'flavors': fakes.FAKE_FLAVOR_LIST})]
    self.register_uris(uris_to_mock)
    flavors = self.cloud.list_flavors()
    found = False
    for flavor in flavors:
        if flavor['name'] == 'vanilla':
            found = True
            break
    self.assertTrue(found)
    needed_keys = {'name', 'ram', 'vcpus', 'id', 'is_public', 'disk'}
    if found:
        self.assertTrue(needed_keys.issubset(flavor.keys()))
    self.assert_calls()