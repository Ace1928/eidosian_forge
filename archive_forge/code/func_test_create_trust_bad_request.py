import datetime
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_trust_bad_request(self):
    self.post('/OS-TRUST/trusts', body={'trust': {}}, expected_status=http.client.FORBIDDEN)