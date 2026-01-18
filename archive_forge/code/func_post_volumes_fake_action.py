from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_volumes_fake_action(self, body, **kw):
    _body = None
    resp = 202
    return (resp, {}, _body)