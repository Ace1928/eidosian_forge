from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def put_backups_1234(self, **kw):
    backup = fakes_base._stub_backup(id='1234', base_uri='http://localhost:8776', tenant_id='0fa851f6668144cf9cd8c8419c1646c1')
    return (200, {}, {'backups': backup})