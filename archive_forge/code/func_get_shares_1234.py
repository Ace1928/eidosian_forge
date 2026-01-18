import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_shares_1234(self, **kw):
    share = {'share': {'id': 1234, 'name': 'sharename', 'status': 'available'}}
    return (200, {}, share)