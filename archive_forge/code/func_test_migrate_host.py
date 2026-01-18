from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
def test_migrate_host(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.0'))
    v = cs.volumes.get('1234')
    self._assert_request_id(v)
    vol = cs.volumes.migrate_volume(v, 'host_dest', False, False)
    cs.assert_called('POST', '/volumes/1234/action', {'os-migrate_volume': {'host': 'host_dest', 'force_host_copy': False, 'lock_volume': False}})
    self._assert_request_id(vol)