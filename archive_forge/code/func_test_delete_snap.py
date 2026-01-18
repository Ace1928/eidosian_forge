from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_delete_snap(self):
    vs = self.cs.assisted_volume_snapshots.delete('x', {})
    self.assert_request_id(vs, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('DELETE', '/os-assisted-volume-snapshots/x?delete_info={}')