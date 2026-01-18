from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.v2 import services
def test_list_services_with_two_search_opts(self):
    manager = self._get_manager('2.7')
    host = 'fake_host'
    binary = 'fake_binary'
    query_string = '?binary=%s&host=%s' % (binary, host)
    with mock.patch.object(manager, '_list', mock.Mock(return_value=None)):
        manager.list({'binary': binary, 'host': host})
        manager._list.assert_called_once_with(services.RESOURCE_PATH + query_string, services.RESOURCE_NAME)