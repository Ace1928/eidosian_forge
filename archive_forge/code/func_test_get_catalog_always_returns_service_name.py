import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_get_catalog_always_returns_service_name(self):
    user_id = uuid.uuid4().hex
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    project = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project['id'], project)
    named_svc = unit.new_service_ref()
    PROVIDERS.catalog_api.create_service(named_svc['id'], named_svc)
    self.create_endpoint(service_id=named_svc['id'])
    unnamed_svc = unit.new_service_ref(name=None)
    del unnamed_svc['name']
    PROVIDERS.catalog_api.create_service(unnamed_svc['id'], unnamed_svc)
    self.create_endpoint(service_id=unnamed_svc['id'])
    catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, project['id'])
    named_endpoint = [ep for ep in catalog if ep['type'] == named_svc['type']][0]
    self.assertEqual(named_svc['name'], named_endpoint['name'])
    unnamed_endpoint = [ep for ep in catalog if ep['type'] == unnamed_svc['type']][0]
    self.assertEqual('', unnamed_endpoint['name'])