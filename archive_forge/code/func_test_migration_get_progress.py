from unittest import mock
import ddt
from manilaclient import base
from manilaclient.common import constants
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_servers
def test_migration_get_progress(self):
    share_server = 'fake_share_server'
    returned = 'fake'
    with mock.patch.object(self.manager, '_action', mock.Mock(return_value=['200', returned])):
        result = self.manager.migration_get_progress(share_server)
        self.manager._action.assert_called_once_with('migration_get_progress', share_server)
        self.assertEqual(returned, result)