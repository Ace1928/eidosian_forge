from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_all_availability_zone_names(self):
    self.register_uris([dict(method='GET', uri='{endpoint}/os-availability-zone'.format(endpoint=fakes.COMPUTE_ENDPOINT), json=_fake_zone_list)])
    self.assertEqual(['az1', 'nova'], self.cloud.list_availability_zone_names(unavailable=True))
    self.assert_calls()