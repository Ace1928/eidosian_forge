from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_os_volume_manage_detail(self, **kw):
    vol_id = 'volume-ffffffff-0000-ffff-0000-ffffffffffff'
    vols = [{'size': 4, 'reason_not_safe': 'volume in use', 'safe_to_manage': False, 'extra_info': 'qos_setting:high', 'reference': {'source-name': vol_id}, 'actual_size': 4.0}, {'size': 5, 'reason_not_safe': None, 'safe_to_manage': True, 'extra_info': 'qos_setting:low', 'actual_size': 4.3, 'reference': {'source-name': 'myvol'}}]
    return (200, {}, {'manageable-volumes': vols})