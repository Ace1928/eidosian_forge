from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_backups_import_record(self, **kw):
    base_uri = 'http://localhost:8776'
    tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
    backup1 = '76a17945-3c6f-435c-975b-b5685db10b62'
    return (200, {}, {'backup': _stub_backup(backup1, base_uri, tenant_id)})