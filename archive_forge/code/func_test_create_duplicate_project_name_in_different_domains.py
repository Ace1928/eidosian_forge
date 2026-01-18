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
def test_create_duplicate_project_name_in_different_domains(self):
    new_domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
    project1 = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project2 = unit.new_project_ref(name=project1['name'], domain_id=new_domain['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    PROVIDERS.resource_api.create_project(project2['id'], project2)