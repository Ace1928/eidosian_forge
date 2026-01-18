from unittest import mock
from manilaclient import base
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
def test_findall_with_all_tenants(self):
    cs.shares.list = mock.Mock(return_value=[])
    cs.shares.findall()
    cs.shares.list.assert_called_with(search_opts={'all_tenants': 1, 'is_soft_deleted': True})