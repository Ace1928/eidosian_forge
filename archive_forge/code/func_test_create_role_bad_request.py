import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_role_bad_request(self):
    """Call ``POST /roles``."""
    self.post('/roles', body={'role': {}}, expected_status=http.client.BAD_REQUEST)