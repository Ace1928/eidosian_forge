from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def post_group_snapshots(self, **kw):
    group_snap = _stub_group_snapshot()
    return (202, {}, {'group_snapshot': group_snap})