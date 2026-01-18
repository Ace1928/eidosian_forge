from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def put_snapshots_1234(self, **kw):
    snapshot = _stub_snapshot(id='1234')
    snapshot.update(kw['body']['snapshot'])
    return (200, {}, {'snapshot': snapshot})