import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_endpoints_associated_with_project_endpoint_group(self):
    """GET & HEAD /OS-EP-FILTER/projects/{project_id}/endpoints.

        Valid project, endpoint id, and endpoint group test case.

        """
    service_ref = unit.new_service_ref()
    response = self.post('/services', body={'service': service_ref})
    service_id2 = response.result['service']['id']
    self._create_endpoint_and_associations(self.default_domain_project_id, service_id2)
    self._create_endpoint_and_associations(self.default_domain_project_id)
    self.put(self.default_request_url)
    body = copy.deepcopy(self.DEFAULT_ENDPOINT_GROUP_BODY)
    body['endpoint_group']['filters'] = {'service_id': service_id2}
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, body)
    self._create_endpoint_group_project_association(endpoint_group_id, self.default_domain_project_id)
    endpoints_url = '/OS-EP-FILTER/projects/%(project_id)s/endpoints' % {'project_id': self.default_domain_project_id}
    r = self.get(endpoints_url, expected_status=http.client.OK)
    endpoints = self.assertValidEndpointListResponse(r)
    self.assertEqual(2, len(endpoints))
    self.head(endpoints_url, expected_status=http.client.OK)
    user_id = uuid.uuid4().hex
    catalog_list = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
    self.assertEqual(2, len(catalog_list))
    url = self._get_project_endpoint_group_url(endpoint_group_id, self.default_domain_project_id)
    self.delete(url)
    url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
    self.delete(url)
    r = self.get(endpoints_url)
    endpoints = self.assertValidEndpointListResponse(r)
    self.assertEqual(1, len(endpoints))
    catalog_list = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.default_domain_project_id)
    self.assertEqual(1, len(catalog_list))