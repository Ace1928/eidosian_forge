import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_replicas_detail(self, **kw):
    replicas = {'share_replicas': [self.fake_share_replica]}
    return (200, {}, replicas)