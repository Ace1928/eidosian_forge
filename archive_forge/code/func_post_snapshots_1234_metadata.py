from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_snapshots_1234_metadata(self, **kw):
    return (200, {}, {'metadata': {'key1': 'val1', 'key2': 'val2'}})