from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_group_snapshots_detail(self, **kw):
    return (200, {}, {'group_snapshots': [_stub_group_snapshot(id='1234'), _stub_group_snapshot(id='4567')]})