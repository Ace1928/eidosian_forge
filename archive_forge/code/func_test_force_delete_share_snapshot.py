from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
@ddt.data(('2.6', type('SnapshotUUID', (object,), {'uuid': '1234'})), ('2.6', '1234'), ('2.7', type('SnapshotUUID', (object,), {'uuid': '1234'})), ('2.7', '1234'))
@ddt.unpack
def test_force_delete_share_snapshot(self, microversion, snapshot):
    manager = self._get_manager(microversion)
    if api_versions.APIVersion(microversion) > api_versions.APIVersion('2.6'):
        action_name = 'force_delete'
    else:
        action_name = 'os-force_delete'
    with mock.patch.object(manager, '_action', mock.Mock()):
        manager.force_delete(snapshot)
        manager._action.assert_called_once_with(action_name, '1234')