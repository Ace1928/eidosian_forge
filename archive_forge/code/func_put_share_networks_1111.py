import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def put_share_networks_1111(self, **kwargs):
    share_network = {'share_network': {'id': 1111}}
    return (200, {}, share_network)