import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_group_snapshot(self):
    lst = cs.group_snapshots.list()
    cs.assert_called('GET', '/group_snapshots/detail')
    self._assert_request_id(lst)