import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_transfers_detail(self, **kw):
    transfer = {'transfers': [self.fake_transfer]}
    return (202, {}, transfer)