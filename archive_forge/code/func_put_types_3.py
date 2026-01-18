from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def put_types_3(self, **kw):
    return (200, {}, {'volume_type': {'id': 3, 'name': 'test-type-2', 'description': 'test_type-3-desc', 'is_public': True, 'extra_specs': {}}})