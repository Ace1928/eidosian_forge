import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_create_delete_share(self):
    name = data_utils.rand_name('autotest_share_name')
    create = self.create_share(self.protocol, name=name, client=self.user_client)
    self.assertEqual('creating', create['status'])
    self.assertEqual(name, create['name'])
    self.assertEqual('1', create['size'])
    self.assertEqual(self.protocol.upper(), create['share_proto'])
    self.user_client.delete_share(create['id'])
    self.user_client.wait_for_share_deletion(create['id'])