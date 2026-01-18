from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_types_1_encryption(self, **kw):
    return (200, {}, {'id': 1, 'volume_type_id': 1, 'provider': 'test', 'cipher': 'test', 'key_size': 1, 'control_location': 'front-end'})