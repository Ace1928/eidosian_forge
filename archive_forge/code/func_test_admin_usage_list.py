from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_admin_usage_list(self):
    self.nova('usage-list')