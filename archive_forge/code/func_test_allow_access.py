from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
def test_allow_access(self):
    snapshot = 'fake_snapshot'
    access_type = 'fake_type'
    access_to = 'fake_to'
    access = ('foo', {'snapshot_access': 'fake'})
    version = api_versions.APIVersion('2.32')
    mock_microversion = mock.Mock(api_version=version)
    manager = share_snapshots.ShareSnapshotManager(api=mock_microversion)
    with mock.patch.object(manager, '_action', mock.Mock(return_value=access)):
        result = manager.allow(snapshot, access_type, access_to)
        self.assertEqual('fake', result)
        manager._action.assert_called_once_with('allow_access', snapshot, {'access_type': access_type, 'access_to': access_to})