import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def delete_share_networks_1234_subnets_fake_subnet1(self, **kwargs):
    return (202, {}, None)