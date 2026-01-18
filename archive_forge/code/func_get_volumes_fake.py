from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_volumes_fake(self, **kw):
    r = {'volume': self.get_volumes_detail(id='fake')[2]['volumes'][0]}
    return (200, {}, r)