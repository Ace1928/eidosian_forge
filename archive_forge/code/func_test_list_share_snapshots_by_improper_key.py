from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
def test_list_share_snapshots_by_improper_key(self):
    self.assertRaises(ValueError, cs.share_snapshots.list, sort_key='fake')