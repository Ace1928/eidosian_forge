import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def test_attribute_presented(self):
    server = self._create_server()
    self._show_server_and_check_lock_attr(server, False)
    self.nova('lock %s' % server.id)
    self._show_server_and_check_lock_attr(server, True)
    self.nova('unlock %s' % server.id)
    self._show_server_and_check_lock_attr(server, False)