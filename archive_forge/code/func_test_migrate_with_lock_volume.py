from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_migrate_with_lock_volume(self):
    v = cs.volumes.get('1234')
    self._assert_request_id(v)
    vol = cs.volumes.migrate_volume(v, 'dest', False, True)
    cs.assert_called('POST', '/volumes/1234/action', {'os-migrate_volume': {'host': 'dest', 'force_host_copy': False, 'lock_volume': True}})
    self._assert_request_id(vol)