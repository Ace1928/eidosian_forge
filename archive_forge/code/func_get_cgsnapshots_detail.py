from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_cgsnapshots_detail(self, **kw):
    return (200, {}, {'cgsnapshots': [_stub_cgsnapshot(id='1234'), _stub_cgsnapshot(id='4567')]})