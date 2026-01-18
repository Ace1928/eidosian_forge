import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_group_snapshot_with_group_id(self):
    snap = cs.group_snapshots.create('1234')
    expected = {'group_snapshot': {'description': None, 'name': None, 'group_id': '1234'}}
    cs.assert_called('POST', '/group_snapshots', body=expected)
    self._assert_request_id(snap)