import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_networks_1234_subnets_fake_subnet_id(self, **kw):
    subnet = {'share_network_subnet': {'id': 'fake_subnet_id'}}
    return (200, {}, subnet)