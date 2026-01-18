import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_quota_sets_1234_detail(self, *args, **kwargs):
    quota_set = {'quota_set': {'id': '1234', 'shares': {'in_use': 0, 'limit': 50, 'reserved': 0}, 'gigabytes': {'in_use': 0, 'limit': 10000, 'reserved': 0}, 'snapshots': {'in_use': 0, 'limit': 50, 'reserved': 0}, 'snapshot_gigabytes': {'in_use': 0, 'limit': 1000, 'reserved': 0}, 'share_networks': {'in_use': 0, 'limit': 10, 'reserved': 0}}}
    return (200, {}, quota_set)