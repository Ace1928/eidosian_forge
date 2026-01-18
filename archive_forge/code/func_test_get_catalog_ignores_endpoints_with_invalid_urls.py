import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_get_catalog_ignores_endpoints_with_invalid_urls(self):
    user_id = uuid.uuid4().hex
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    project = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project['id'], project)
    catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, project['id'])
    self.assertEqual(1, len(catalog[0]['endpoints']))
    self.assertEqual(1, len(PROVIDERS.catalog_api.list_endpoints()))
    self.create_endpoint(self.service_id, url='http://keystone/%(project_id)')
    self.create_endpoint(self.service_id, url='http://keystone/%(you_wont_find_me)s')
    catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, project['id'])
    self.assertEqual(1, len(catalog[0]['endpoints']))
    self.assertEqual(3, len(PROVIDERS.catalog_api.list_endpoints()))
    self.create_endpoint(self.service_id, url='http://keystone/%(project_id)s')
    catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, project['id'])
    self.assertThat(catalog[0]['endpoints'], matchers.HasLength(2))
    project_id = None
    catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, project_id)
    self.assertThat(catalog[0]['endpoints'], matchers.HasLength(1))