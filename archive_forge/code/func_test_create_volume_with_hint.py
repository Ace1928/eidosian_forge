from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
def test_create_volume_with_hint(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.0'))
    vol = cs.volumes.create(1, scheduler_hints='uuid')
    expected = {'volume': {'description': None, 'availability_zone': None, 'source_volid': None, 'snapshot_id': None, 'size': 1, 'name': None, 'imageRef': None, 'volume_type': None, 'metadata': {}, 'consistencygroup_id': None, 'backup_id': None}, 'OS-SCH-HNT:scheduler_hints': 'uuid'}
    cs.assert_called('POST', '/volumes', body=expected)
    self._assert_request_id(vol)