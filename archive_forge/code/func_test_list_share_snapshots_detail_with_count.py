from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
def test_list_share_snapshots_detail_with_count(self):
    search_opts = {'with_count': 'True'}
    snapshots, count = cs.share_snapshots.list(detailed=True, search_opts=search_opts)
    cs.assert_called('GET', '/snapshots/detail?with_count=True')
    self.assertEqual(2, count)
    self.assertEqual(1, len(snapshots))