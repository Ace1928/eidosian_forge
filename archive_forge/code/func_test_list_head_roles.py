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
def test_list_head_roles(self):
    """Call ``GET & HEAD /roles``."""
    resource_url = '/roles'
    r = self.get(resource_url)
    self.assertValidRoleListResponse(r, ref=self.role, resource_url=resource_url)
    self.head(resource_url, expected_status=http.client.OK)