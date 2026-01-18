import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_shares_1234_metadata(self, **kw):
    return (204, {}, {'metadata': {'test_key': 'test_value'}})