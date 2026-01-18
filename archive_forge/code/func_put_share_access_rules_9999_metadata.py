import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def put_share_access_rules_9999_metadata(self, **kw):
    return (200, {}, {'metadata': {'key1': 'v1', 'key2': 'v2'}})