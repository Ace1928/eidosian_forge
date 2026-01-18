import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
def test_list_shares_by_inexact_unicode_option(self):
    self.create_share(name=u'共享名称', description=u'共享描述', client=self.user_client)
    filters = {'name~': u'名称'}
    shares = self.user_client.list_shares(filters=filters)
    self.assertGreater(len(shares), 0)
    filters = {'description~': u'描述'}
    shares = self.user_client.list_shares(filters=filters)
    self.assertGreater(len(shares), 0)