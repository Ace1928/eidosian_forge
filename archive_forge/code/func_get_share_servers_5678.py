import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_servers_5678(self, **kw):
    share_servers = {'share_server': {'id': 5678, 'share_network_id': 'fake_network_id_2'}}
    return (200, {}, share_servers)