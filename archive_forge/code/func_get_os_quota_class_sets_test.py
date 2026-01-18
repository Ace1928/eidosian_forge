from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_os_quota_class_sets_test(self, **kw):
    return (200, {}, {'quota_class_set': {'class_name': 'test', 'volumes': 1, 'snapshots': 1, 'gigabytes': 1, 'backups': 1, 'backup_gigabytes': 1, 'per_volume_gigabytes': 1}})