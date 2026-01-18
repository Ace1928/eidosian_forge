from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_types_1(self, **kw):
    return (200, {}, {'volume_type': {'id': 1, 'name': 'test-type-1', 'description': 'test_type-1-desc', 'extra_specs': {'key': 'value'}}})