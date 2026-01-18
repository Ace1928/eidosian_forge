from unittest import mock
import ddt
from manilaclient import base
from manilaclient.common import constants
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_servers
def test_get_nonexistent_share_network_name(self):
    resource_class = share_servers.ShareServer(manager=self, info={})
    try:
        resource_class.share_network_name
    except AttributeError:
        pass
    else:
        raise Exception("Expected exception 'AttributeError' has not been raised.")