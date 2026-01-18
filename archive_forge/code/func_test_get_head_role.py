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
def test_get_head_role(self):
    """Call ``GET & HEAD /roles/{role_id}``."""
    resource_url = '/roles/%(role_id)s' % {'role_id': self.role_id}
    r = self.get(resource_url)
    self.assertValidRoleResponse(r, self.role)
    self.head(resource_url, expected_status=http.client.OK)