from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_snapshots as snapshots
def test_create_minimal_args(self):
    fake_share_group_snapshot = fake.ShareGroupSnapshot()
    mock_create = self.mock_object(self.manager, '_create', mock.Mock(return_value=fake_share_group_snapshot))
    result = self.manager.create(fake.ShareGroupSnapshot)
    self.assertIs(fake_share_group_snapshot, result)
    expected_body = {snapshots.RESOURCE_NAME: {'share_group_id': fake.ShareGroupSnapshot().id}}
    mock_create.assert_called_once_with(snapshots.RESOURCES_PATH, expected_body, snapshots.RESOURCE_NAME)