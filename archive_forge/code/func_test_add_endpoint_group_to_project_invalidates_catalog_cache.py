import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
@unit.skip_if_cache_disabled('catalog')
def test_add_endpoint_group_to_project_invalidates_catalog_cache(self):
    endpoint_id2 = uuid.uuid4().hex
    endpoint2 = unit.new_endpoint_ref(service_id=self.service_id, region_id=self.region_id, interface='admin', id=endpoint_id2)
    PROVIDERS.catalog_api.create_endpoint(endpoint_id2, endpoint2)
    self.put(self.default_request_url)
    user_id = uuid.uuid4().hex
    catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
    self.assertThat(catalog[0]['endpoints'], matchers.HasLength(1))
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    PROVIDERS.catalog_api.driver.add_endpoint_group_to_project(endpoint_group_id, self.default_domain_project_id)
    invalid_catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
    self.assertThat(invalid_catalog[0]['endpoints'], matchers.HasLength(1))
    self.assertEqual(catalog, invalid_catalog)
    PROVIDERS.catalog_api.driver.remove_endpoint_group_from_project(endpoint_group_id, self.default_domain_project_id)
    PROVIDERS.catalog_api.add_endpoint_group_to_project(endpoint_group_id, self.default_domain_project_id)
    catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
    self.assertThat(catalog[0]['endpoints'], matchers.HasLength(2))
    ep_id_list = [catalog[0]['endpoints'][0]['id'], catalog[0]['endpoints'][1]['id']]
    self.assertCountEqual([self.endpoint_id, endpoint_id2], ep_id_list)