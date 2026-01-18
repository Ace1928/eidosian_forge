import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_snapshots_1234_export_locations(self):
    snapshot_export_location = {'share_snapshot_export_locations': [get_fake_export_location()]}
    return (200, {}, snapshot_export_location)