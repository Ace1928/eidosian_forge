import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
@unit.skip_if_cache_disabled('catalog')
def test_create_endpoint_project_association_invalidates_cache(self):
    endpoint_id2 = uuid.uuid4().hex
    endpoint2 = unit.new_endpoint_ref(service_id=self.service_id, region_id=self.region_id, interface='public', id=endpoint_id2)
    PROVIDERS.catalog_api.create_endpoint(endpoint_id2, endpoint2.copy())
    self.put(self.default_request_url)
    user_id = uuid.uuid4().hex
    catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
    self.assertEqual(1, len(catalog[0]['endpoints']))
    self.assertEqual(self.endpoint_id, catalog[0]['endpoints'][0]['id'])
    PROVIDERS.catalog_api.driver.add_endpoint_to_project(endpoint_id2, self.default_domain_project_id)
    catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
    self.assertEqual(1, len(catalog[0]['endpoints']))
    PROVIDERS.catalog_api.driver.remove_endpoint_from_project(endpoint_id2, self.default_domain_project_id)
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': endpoint_id2})
    catalog = self.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
    self.assertEqual(2, len(catalog[0]['endpoints']))
    ep_id_list = [catalog[0]['endpoints'][0]['id'], catalog[0]['endpoints'][1]['id']]
    self.assertCountEqual([self.endpoint_id, endpoint_id2], ep_id_list)