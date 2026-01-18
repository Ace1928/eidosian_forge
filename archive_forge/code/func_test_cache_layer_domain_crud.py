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
@unit.skip_if_cache_disabled('resource')
def test_cache_layer_domain_crud(self):
    domain = unit.new_domain_ref()
    domain_id = domain['id']
    PROVIDERS.resource_api.create_domain(domain_id, domain)
    project_domain_ref = PROVIDERS.resource_api.get_project(domain_id)
    domain_ref = PROVIDERS.resource_api.get_domain(domain_id)
    updated_project_domain_ref = copy.deepcopy(project_domain_ref)
    updated_project_domain_ref['name'] = uuid.uuid4().hex
    updated_domain_ref = copy.deepcopy(domain_ref)
    updated_domain_ref['name'] = updated_project_domain_ref['name']
    PROVIDERS.resource_api.driver.update_project(domain_id, updated_project_domain_ref)
    self.assertLessEqual(domain_ref.items(), PROVIDERS.resource_api.get_domain(domain_id).items())
    PROVIDERS.resource_api.get_domain.invalidate(PROVIDERS.resource_api, domain_id)
    self.assertLessEqual(updated_domain_ref.items(), PROVIDERS.resource_api.get_domain(domain_id).items())
    PROVIDERS.resource_api.update_domain(domain_id, domain_ref)
    self.assertLessEqual(domain_ref.items(), PROVIDERS.resource_api.get_domain(domain_id).items())
    project_domain_ref_disabled = project_domain_ref.copy()
    project_domain_ref_disabled['enabled'] = False
    PROVIDERS.resource_api.driver.update_project(domain_id, project_domain_ref_disabled)
    PROVIDERS.resource_api.driver.update_project(domain_id, {'enabled': False})
    PROVIDERS.resource_api.driver.delete_project(domain_id)
    self.assertLessEqual(domain_ref.items(), PROVIDERS.resource_api.get_domain(domain_id).items())
    PROVIDERS.resource_api.get_domain.invalidate(PROVIDERS.resource_api, domain_id)
    self.assertRaises(exception.DomainNotFound, PROVIDERS.resource_api.get_domain, domain_id)
    PROVIDERS.resource_api.create_domain(domain_id, domain)
    PROVIDERS.resource_api.get_domain(domain_id)
    domain['enabled'] = False
    PROVIDERS.resource_api.driver.update_project(domain_id, domain)
    PROVIDERS.resource_api.driver.update_project(domain_id, {'enabled': False})
    PROVIDERS.resource_api.delete_domain(domain_id)
    self.assertRaises(exception.DomainNotFound, PROVIDERS.resource_api.get_domain, domain_id)