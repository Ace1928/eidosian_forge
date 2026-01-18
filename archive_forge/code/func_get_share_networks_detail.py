import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_networks_detail(self, **kw):
    share_nw = {'share_networks': [{'id': 1234, 'name': 'fake_share_nw'}, {'id': 4321, 'name': 'duplicated_name'}, {'id': 4322, 'name': 'duplicated_name'}]}
    return (200, {}, share_nw)