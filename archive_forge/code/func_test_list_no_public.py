from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_types as types
def test_list_no_public(self):
    fake_share_group_type = fake.ShareGroupType()
    mock_list = self.mock_object(self.manager, '_list', mock.Mock(return_value=[fake_share_group_type]))
    result = self.manager.list(show_all=False)
    self.assertEqual([fake_share_group_type], result)
    mock_list.assert_called_once_with(types.RESOURCES_PATH, types.RESOURCES_NAME)