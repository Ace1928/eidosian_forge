import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def delete_share_group_snapshots_1234(self, **kw):
    return (202, {}, None)