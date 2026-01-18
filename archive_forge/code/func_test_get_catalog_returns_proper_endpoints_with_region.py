import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_get_catalog_returns_proper_endpoints_with_region(self):
    service = unit.new_service_ref()
    service_id = service['id']
    PROVIDERS.catalog_api.create_service(service_id, service)
    endpoint = unit.new_endpoint_ref(service_id=service_id)
    region = unit.new_region_ref(id=endpoint['region_id'])
    PROVIDERS.catalog_api.create_region(region)
    PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
    endpoint = PROVIDERS.catalog_api.get_endpoint(endpoint['id'])
    user_id = uuid.uuid4().hex
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    project = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project['id'], project)
    catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, project['id'])
    self.assertValidCatalogEndpoint(catalog[0]['endpoints'][0], ref=endpoint)