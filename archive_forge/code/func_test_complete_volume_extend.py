from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
@ddt.data(False, True)
def test_complete_volume_extend(self, error):
    cs = fakes.FakeClient(api_versions.APIVersion('3.71'))
    v = cs.volumes.get('1234')
    self._assert_request_id(v)
    vol = cs.volumes.extend_volume_completion(v, error)
    cs.assert_called('POST', '/volumes/1234/action', {'os-extend_volume_completion': {'error': error}})
    self._assert_request_id(vol)