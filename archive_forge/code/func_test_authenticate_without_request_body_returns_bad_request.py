import datetime
import hashlib
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone.common import utils
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_authenticate_without_request_body_returns_bad_request(self):
    self.post('/ec2tokens', expected_status=http.client.BAD_REQUEST)