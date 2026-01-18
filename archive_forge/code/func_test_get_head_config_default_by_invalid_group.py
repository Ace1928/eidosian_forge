import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_config_default_by_invalid_group(self):
    """Call ``GET & HEAD for /domains/config/{bad-group}/default``."""
    self.get('/domains/config/resource/default', expected_status=http.client.FORBIDDEN)
    self.head('/domains/config/resource/default', expected_status=http.client.FORBIDDEN)
    url = '/domains/config/%s/default' % uuid.uuid4().hex
    self.get(url, expected_status=http.client.FORBIDDEN)
    self.head(url, expected_status=http.client.FORBIDDEN)