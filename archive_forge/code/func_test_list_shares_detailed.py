from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
def test_list_shares_detailed(self):
    search_opts = {'with_count': 'True'}
    shares, count = cs.shares.list(detailed=True, search_opts=search_opts)
    cs.assert_called('GET', '/shares/detail?is_public=True&with_count=True')
    self.assertEqual(2, count)
    self.assertEqual(1, len(shares))