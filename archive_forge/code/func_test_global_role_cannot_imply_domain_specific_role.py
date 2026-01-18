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
def test_global_role_cannot_imply_domain_specific_role(self):
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    domain_role_ref = unit.new_role_ref(domain_id=domain['id'])
    domain_role = PROVIDERS.role_api.create_role(domain_role_ref['id'], domain_role_ref)
    global_role_ref = unit.new_role_ref()
    global_role = PROVIDERS.role_api.create_role(global_role_ref['id'], global_role_ref)
    self.put('/roles/%s/implies/%s' % (global_role['id'], domain_role['id']), expected_status=http.client.FORBIDDEN)