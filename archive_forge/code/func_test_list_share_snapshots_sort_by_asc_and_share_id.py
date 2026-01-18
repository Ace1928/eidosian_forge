from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
def test_list_share_snapshots_sort_by_asc_and_share_id(self):
    cs.share_snapshots.list(detailed=False, sort_key='share_id', sort_dir='asc')
    cs.assert_called('GET', '/snapshots?sort_dir=asc&sort_key=share_id')