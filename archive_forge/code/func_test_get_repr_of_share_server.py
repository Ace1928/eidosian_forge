from unittest import mock
import ddt
from manilaclient import base
from manilaclient.common import constants
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_servers
def test_get_repr_of_share_server(self):
    self.assertIn('ShareServer: %s' % self.share_server_id, repr(self.resource_class))