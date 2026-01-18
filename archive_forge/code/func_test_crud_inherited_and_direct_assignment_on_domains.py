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
def test_crud_inherited_and_direct_assignment_on_domains(self):
    self._test_crud_inherited_and_direct_assignment_on_target('/domains/%s' % self.domain_id)