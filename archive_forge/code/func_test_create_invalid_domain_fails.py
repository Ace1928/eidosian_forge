import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.resource.backends import sql as resource_sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import utils as test_utils
def test_create_invalid_domain_fails(self):
    new_group = unit.new_group_ref(domain_id='doesnotexist')
    self.assertRaises(exception.DomainNotFound, PROVIDERS.identity_api.create_group, new_group)
    new_user = unit.new_user_ref(domain_id='doesnotexist')
    self.assertRaises(exception.DomainNotFound, PROVIDERS.identity_api.create_user, new_user)