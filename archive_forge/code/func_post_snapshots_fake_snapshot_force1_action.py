import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_snapshots_fake_snapshot_force1_action(self, **kwargs):
    return (202, {}, None)