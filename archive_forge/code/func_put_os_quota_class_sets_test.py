from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def put_os_quota_class_sets_test(self, body, **kw):
    assert list(body) == ['quota_class_set']
    fakes.assert_has_keys(body['quota_class_set'])
    return (200, {}, {'quota_class_set': {'volumes': 2, 'snapshots': 2, 'gigabytes': 1, 'backups': 1, 'backup_gigabytes': 1, 'per_volume_gigabytes': 1}})