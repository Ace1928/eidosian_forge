from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
def test_access_list(self):
    cs.share_snapshots.access_list(1234)
    cs.assert_called('GET', '/snapshots/1234/access-list')