import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_group_types_2(self, **kw):
    share_group_type = {'share_type': {'id': 2, 'name': 'test-group-type-2', 'group_specs': {'key2': 'value2'}, 'share_types': ['type3', 'type4'], 'is_public': True}}
    return (200, {}, share_group_type)