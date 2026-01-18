from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_flavor(self):
    self.use_compute_discovery()
    self.register_uris([dict(method='GET', uri='{endpoint}/flavors/vanilla'.format(endpoint=fakes.COMPUTE_ENDPOINT), json=fakes.FAKE_FLAVOR), dict(method='DELETE', uri='{endpoint}/flavors/{id}'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=fakes.FLAVOR_ID))])
    self.assertTrue(self.cloud.delete_flavor('vanilla'))
    self.assert_calls()