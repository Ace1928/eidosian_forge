import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_types_1234(self, **kw):
    return (200, {}, {'share_type': {'id': 1234, 'name': 'test-type-1234', 'share_type_access:is_public': True, 'description': 'test share type desc', 'extra_specs': {'test': 'test'}, 'required_extra_specs': {'test': 'test'}}})