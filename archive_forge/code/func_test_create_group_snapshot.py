import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_group_snapshot(self):
    snap = cs.group_snapshots.create('group_snap')
    cs.assert_called('POST', '/group_snapshots')
    self._assert_request_id(snap)