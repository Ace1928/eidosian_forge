import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_groups_detail(self, **kw):
    share_groups = {'share_groups': [self.fake_share_group]}
    return (200, {}, share_groups)