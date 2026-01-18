from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def put_volumes_1234(self, **kw):
    volume = _stub_volume(id='1234')
    volume.update(kw['body']['volume'])
    return (200, {}, {'volume': volume})