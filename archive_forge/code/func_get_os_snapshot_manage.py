from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_os_snapshot_manage(self, **kw):
    snap_id = 'snapshot-ffffffff-0000-ffff-0000-ffffffffffff'
    snaps = [{'actual_size': 4.0, 'size': 4, 'safe_to_manage': False, 'source_id_type': 'source-name', 'source_cinder_id': '00000000-ffff-0000-ffff-00000000', 'reference': {'source-name': snap_id}, 'source_identifier': 'volume-00000000-ffff-0000-ffff-000000'}, {'actual_size': 4.3, 'reference': {'source-name': 'mysnap'}, 'source_id_type': 'source-name', 'source_identifier': 'myvol', 'safe_to_manage': True, 'source_cinder_id': None, 'size': 5}]
    return (200, {}, {'manageable-snapshots': snaps})