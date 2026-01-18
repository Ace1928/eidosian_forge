from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshot_instances
@ddt.data(True, False)
def test_list_with_snapshot(self, detailed):
    if detailed:
        url = '/snapshot-instances/detail'
    else:
        url = '/snapshot-instances'
    self.mock_object(self.manager, '_list', mock.Mock())
    self.manager.list(detailed=detailed, snapshot='snapshot_id')
    self.manager._list.assert_called_once_with(url + '?snapshot_id=snapshot_id', 'snapshot_instances')