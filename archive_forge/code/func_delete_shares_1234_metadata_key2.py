import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def delete_shares_1234_metadata_key2(self, **kw):
    return (204, {}, None)