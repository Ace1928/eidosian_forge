import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_get_group_snapshot(self):
    group_snapshot_id = '1234'
    snap = cs.group_snapshots.get(group_snapshot_id)
    cs.assert_called('GET', '/group_snapshots/%s' % group_snapshot_id)
    self._assert_request_id(snap)