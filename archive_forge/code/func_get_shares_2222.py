import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_shares_2222(self, **kw):
    share = {'share': {'id': 2222, 'name': 'sharename'}}
    return (200, {}, share)