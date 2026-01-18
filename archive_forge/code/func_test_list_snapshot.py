from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshot_export_locations
def test_list_snapshot(self):
    snapshot_id = '1234'
    cs.share_snapshot_export_locations.list(snapshot_id, search_opts=None)
    cs.assert_called('GET', '/snapshots/%s/export-locations' % snapshot_id)