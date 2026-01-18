from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_backups
def test_list_with_share(self):
    with mock.patch.object(self.manager, '_list', mock.Mock()):
        self.manager.list(search_opts={'share_id': 'fake_share_id'})
        share_uri = '?share_id=fake_share_id'
        self.manager._list.assert_called_once_with(share_backups.RESOURCES_PATH + '/detail' + share_uri, share_backups.RESOURCES_NAME)