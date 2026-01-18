from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_volumes_detail(self, **kw):
    if kw.get('with_count', False):
        return (200, {}, {'volumes': [_stub_volume(id=kw.get('id', 1234))], 'count': 1})
    return (200, {}, {'volumes': [_stub_volume(id=kw.get('id', 1234))]})