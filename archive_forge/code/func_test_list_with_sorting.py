from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_snapshots as snapshots
def test_list_with_sorting(self):
    fake_share_group_snapshot = fake.ShareGroupSnapshot()
    mock_list = self.mock_object(self.manager, '_list', mock.Mock(return_value=[fake_share_group_snapshot]))
    result = self.manager.list(detailed=False, sort_dir='asc', sort_key='name')
    self.assertEqual([fake_share_group_snapshot], result)
    expected_path = snapshots.RESOURCES_PATH + '?sort_dir=asc&sort_key=name'
    mock_list.assert_called_once_with(expected_path, snapshots.RESOURCES_NAME)