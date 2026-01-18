from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
def test_delete_share_snapshot(self):
    snapshot = cs.share_snapshots.get(1234)
    cs.share_snapshots.delete(snapshot)
    cs.assert_called('DELETE', '/snapshots/1234')