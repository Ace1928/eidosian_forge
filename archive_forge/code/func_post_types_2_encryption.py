from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_types_2_encryption(self, body, **kw):
    return (200, {}, {'encryption': body})