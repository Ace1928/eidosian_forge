from unittest import mock
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import scheduler_stats
@mock.patch.object(scheduler_stats.PoolManager, '_list', mock.Mock())
def test_list_detail_with_two_search_opts(self):
    host = 'fake_host'
    backend = 'fake_backend'
    query_string = '?backend=%s&host=%s' % (backend, host)
    self.manager.list(search_opts={'host': host, 'backend': backend})
    self.manager._list.assert_called_once_with(scheduler_stats.RESOURCES_PATH + '/detail' + query_string, scheduler_stats.RESOURCES_NAME)