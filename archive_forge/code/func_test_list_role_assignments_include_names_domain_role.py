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
def test_list_role_assignments_include_names_domain_role(self):
    role = unit.new_role_ref(domain_id=self.domain['id'])
    PROVIDERS.role_api.create_role(role['id'], role)
    self._test_list_role_assignments_include_names(role)