from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_snapshots_1234(self, **kw):
    return (200, {}, {'snapshot': _stub_snapshot(id='1234')})