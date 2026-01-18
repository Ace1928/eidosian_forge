from openstack.tests import fakes
from openstack.tests.unit import base
def test_cleanup_floating_ips(self):
    self.assertFalse(self.cloud.delete_unattached_floating_ips())