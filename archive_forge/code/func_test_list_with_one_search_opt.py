from unittest import mock
import ddt
from manilaclient import base
from manilaclient.common import constants
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_servers
def test_list_with_one_search_opt(self):
    host = 'fake_host'
    query_string = '?host=%s' % host
    with mock.patch.object(self.manager, '_list', mock.Mock(return_value=None)):
        self.manager.list({'host': host})
        self.manager._list.assert_called_once_with(share_servers.RESOURCES_PATH + query_string, share_servers.RESOURCES_NAME)