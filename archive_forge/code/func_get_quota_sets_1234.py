import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_quota_sets_1234(self, *args, **kwargs):
    quota_set = {'quota_set': {'id': '1234', 'shares': 50, 'gigabytes': 1000, 'snapshots': 50, 'snapshot_gigabytes': 1000, 'share_networks': 10}}
    return (200, {}, quota_set)