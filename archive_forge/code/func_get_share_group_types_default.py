import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_group_types_default(self, **kw):
    return self.get_share_group_types_1(**kw)