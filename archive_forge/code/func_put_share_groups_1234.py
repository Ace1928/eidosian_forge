import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def put_share_groups_1234(self, **kwargs):
    share_group = {'share_group': self.fake_share_group}
    return (200, {}, share_group)