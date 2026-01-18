from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_admin_keypair_list(self):
    self.nova('keypair-list')