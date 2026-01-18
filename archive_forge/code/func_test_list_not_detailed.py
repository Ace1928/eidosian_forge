import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_networks
def test_list_not_detailed(self):
    with mock.patch.object(self.manager, '_list', mock.Mock(return_value=None)):
        self.manager.list(detailed=False)
        self.manager._list.assert_called_once_with(share_networks.RESOURCES_PATH, share_networks.RESOURCES_NAME)