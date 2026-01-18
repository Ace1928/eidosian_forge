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
@unit.skip_if_no_multiple_domains_support
def test_cannot_delete_disabled_domain_with_immutable_project(self):
    domain_id = uuid.uuid4().hex
    domain = {'name': uuid.uuid4().hex, 'id': domain_id, 'is_domain': True}
    PROVIDERS.resource_api.create_domain(domain_id, domain)
    project = unit.new_project_ref(domain_id)
    project['options'][ro_opt.IMMUTABLE_OPT.option_name] = True
    PROVIDERS.resource_api.create_project(project['id'], project)
    PROVIDERS.resource_api.update_domain(domain_id, {'enabled': False})
    self.assertRaises(exception.ResourceDeleteForbidden, PROVIDERS.resource_api.delete_domain, domain_id)