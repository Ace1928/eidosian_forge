import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@ddt.data({'detailed': True, 'url': '/group_snapshots/detail'}, {'detailed': False, 'url': '/group_snapshots'})
@ddt.unpack
def test_list_group_snapshot_detailed(self, detailed, url):
    lst = cs.group_snapshots.list(detailed=detailed)
    cs.assert_called('GET', url)
    self._assert_request_id(lst)