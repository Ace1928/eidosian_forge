import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@ddt.data({'foo': 'bar'}, {'foo': 'bar', '123': None})
def test_list_group_snapshot_with_search_opts(self, opts):
    lst = cs.group_snapshots.list(search_opts=opts)
    cs.assert_called('GET', '/group_snapshots/detail?foo=bar')
    self._assert_request_id(lst)