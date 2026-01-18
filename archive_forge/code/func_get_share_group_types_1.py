import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_group_types_1(self, **kw):
    share_group_type = {'share_group_type': {'id': 1, 'name': 'test-group-type-1', 'group_specs': {'key1': 'value1'}, 'share_types': ['type1', 'type2'], 'is_public': True}}
    return (200, {}, share_group_type)