import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_types_3_os_share_type_access(self, **kw):
    return (200, {}, {'share_type_access': [{'share_type_id': '11111111-1111-1111-1111-111111111111', 'project_id': '00000000-0000-0000-0000-000000000000'}]})