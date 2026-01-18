import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_access_rules(self, **kw):
    access = {'access_list': [{'access_level': 'rw', 'state': 'active', 'id': '1122', 'access_type': 'ip', 'access_to': '10.0.0.7', 'metadata': {'key1': 'v1'}}]}
    return (200, {}, access)