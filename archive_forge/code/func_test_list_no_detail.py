from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_snapshots as snapshots
def test_list_no_detail(self):
    fake_share_group_snapshot = fake.ShareGroupSnapshot()
    mock_list = self.mock_object(self.manager, '_list', mock.Mock(return_value=[fake_share_group_snapshot]))
    result = self.manager.list(detailed=False)
    self.assertEqual([fake_share_group_snapshot], result)
    mock_list.assert_called_once_with(snapshots.RESOURCES_PATH, snapshots.RESOURCES_NAME)