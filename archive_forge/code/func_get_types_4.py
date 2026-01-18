import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_types_4(self, **kw):
    return (200, {}, {'share_type': {'id': 4, 'name': 'test-type-3', 'extra_specs': {}, 'os-share-type-access:is_public': True}})