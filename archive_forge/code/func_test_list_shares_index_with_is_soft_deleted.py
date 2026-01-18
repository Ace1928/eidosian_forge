from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
@ddt.data(True, False)
def test_list_shares_index_with_is_soft_deleted(self, detailed):
    search_opts = {'is_soft_deleted': 'True'}
    cs.shares.list(detailed=detailed, search_opts=search_opts)
    if detailed:
        cs.assert_called('GET', '/shares/detail?is_public=True' + '&is_soft_deleted=True')
    else:
        cs.assert_called('GET', '/shares?is_public=True' + '&is_soft_deleted=True')