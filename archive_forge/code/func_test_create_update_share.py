import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_create_update_share(self):
    name = data_utils.rand_name('autotest_share_name')
    new_name = 'new_' + name
    description = data_utils.rand_name('autotest_share_description')
    new_description = 'new_' + description
    create = self.create_share(self.protocol, name=name, description=description, client=self.user_client)
    self.assertEqual(name, create['name'])
    self.assertEqual(description, create['description'])
    self.assertEqual('False', create['is_public'])
    self.user_client.update_share(create['id'], name=new_name, description=new_description)
    get = self.user_client.get_share(create['id'])
    self.assertEqual(new_name, get['name'])
    self.assertEqual(new_description, get['description'])
    self.assertEqual('False', get['is_public'])
    self.admin_client.update_share(create['id'], is_public=True)
    get = self.user_client.get_share(create['id'])
    self.assertEqual(new_name, get['name'])
    self.assertEqual(new_description, get['description'])
    self.assertEqual('True', get['is_public'])