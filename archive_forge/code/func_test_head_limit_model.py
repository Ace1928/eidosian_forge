import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_head_limit_model(self):
    self.head('/limits/model', expected_status=http.client.OK)