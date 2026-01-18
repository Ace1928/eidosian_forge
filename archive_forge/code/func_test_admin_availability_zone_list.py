from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_admin_availability_zone_list(self):
    self.assertIn('internal', self.nova('availability-zone-list'))