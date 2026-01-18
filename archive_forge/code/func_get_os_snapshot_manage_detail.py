from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_os_snapshot_manage_detail(self, **kw):
    snap_id = 'snapshot-ffffffff-0000-ffff-0000-ffffffffffff'
    snaps = [{'actual_size': 4.0, 'size': 4, 'safe_to_manage': False, 'source_id_type': 'source-name', 'source_cinder_id': '00000000-ffff-0000-ffff-00000000', 'reference': {'source-name': snap_id}, 'source_identifier': 'volume-00000000-ffff-0000-ffff-000000', 'extra_info': 'qos_setting:high', 'reason_not_safe': 'snapshot in use'}, {'actual_size': 4.3, 'reference': {'source-name': 'mysnap'}, 'safe_to_manage': True, 'source_cinder_id': None, 'source_id_type': 'source-name', 'identifier': 'mysnap', 'source_identifier': 'myvol', 'size': 5, 'extra_info': 'qos_setting:low', 'reason_not_safe': None}]
    return (200, {}, {'manageable-snapshots': snaps})