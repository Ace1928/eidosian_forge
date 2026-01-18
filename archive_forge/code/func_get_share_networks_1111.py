import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_networks_1111(self, **kw):
    share_nw = {'share_network': {'id': 1111, 'name': 'fake_share_nw'}}
    return (200, {}, share_nw)