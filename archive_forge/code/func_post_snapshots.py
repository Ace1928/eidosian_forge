from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_snapshots(self, **kw):
    metadata = kw['body']['snapshot'].get('metadata', None)
    snapshot = _stub_snapshot(id='1234', volume_id='1234')
    if snapshot is not None:
        snapshot.update({'metadata': metadata})
    return (202, {}, {'snapshot': snapshot})