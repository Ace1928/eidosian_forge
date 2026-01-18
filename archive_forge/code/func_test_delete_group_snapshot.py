import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_delete_group_snapshot(self):
    s1 = cs.group_snapshots.list()[0]
    snap = s1.delete()
    self._assert_request_id(snap)
    cs.assert_called('DELETE', '/group_snapshots/1234')
    snap = cs.group_snapshots.delete('1234')
    cs.assert_called('DELETE', '/group_snapshots/1234')
    self._assert_request_id(snap)
    snap = cs.group_snapshots.delete(s1)
    cs.assert_called('DELETE', '/group_snapshots/1234')
    self._assert_request_id(snap)