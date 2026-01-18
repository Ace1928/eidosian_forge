from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_admin_timeout(self):
    self.nova('list', flags='--timeout %d' % 60)