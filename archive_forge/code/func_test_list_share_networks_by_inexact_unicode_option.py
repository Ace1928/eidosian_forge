import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_list_share_networks_by_inexact_unicode_option(self):
    self.create_share_network(name=u'网络名称', description=u'网络描述', neutron_net_id='fake_neutron_net_id', neutron_subnet_id='fake_neutron_subnet_id')
    filters = {'name~': u'名称'}
    share_networks = self.admin_client.list_share_networks(filters=filters)
    self.assertGreater(len(share_networks), 0)
    filters = {'description~': u'描述'}
    share_networks = self.admin_client.list_share_networks(filters=filters)
    self.assertGreater(len(share_networks), 0)