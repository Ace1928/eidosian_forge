import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def test_add_many(self):
    uuid = self._boot_server_with_tags()
    self.nova('server-tag-add %s t3 t4' % uuid)
    self.assertEqual(['t1', 't2', 't3', 't4'], self.client.servers.tag_list(uuid))