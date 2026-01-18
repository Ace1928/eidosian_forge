from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_backups_76a17945_3c6f_435c_975b_b5685db10b62_export_record(self, **kw):
    return (200, {}, {'backup-record': {'backup_service': 'fake-backup-service', 'backup_url': 'fake-backup-url'}})