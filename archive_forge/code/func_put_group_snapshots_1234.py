from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def put_group_snapshots_1234(self, **kw):
    return (200, {}, {'group_snapshot': {}})