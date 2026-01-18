import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def put_shares_1234(self, **kwargs):
    share = {'share': {'id': 1234, 'name': 'sharename'}}
    return (200, {}, share)