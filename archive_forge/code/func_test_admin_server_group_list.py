from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_admin_server_group_list(self):
    self.nova('server-group-list')