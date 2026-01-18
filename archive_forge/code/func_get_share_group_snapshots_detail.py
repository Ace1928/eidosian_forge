import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_group_snapshots_detail(self, **kw):
    sg_snapshots = {'share_group_snapshots': [self.fake_share_group_snapshot]}
    return (200, {}, sg_snapshots)