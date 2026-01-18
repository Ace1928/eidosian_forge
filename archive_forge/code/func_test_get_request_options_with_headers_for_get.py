import base64
import copy
from unittest import mock
from urllib import parse as urlparse
from oslo_utils import uuidutils
from osprofiler import _utils as osprofiler_utils
import osprofiler.profiler
from mistralclient.api import httpclient
from mistralclient.tests.unit import base
def test_get_request_options_with_headers_for_get(self):
    m = self.requests_mock.get(EXPECTED_URL, text='text')
    target_auth_url = uuidutils.generate_uuid()
    target_auth_token = uuidutils.generate_uuid()
    target_user_id = 'target_user'
    target_project_id = 'target_project'
    target_service_catalog = 'this should be there'
    target_insecure = 'target insecure'
    target_region = 'target region name'
    target_user_domain_name = 'target user domain name'
    target_project_domain_name = 'target project domain name'
    target_client = httpclient.HTTPClient(API_BASE_URL, auth_token=AUTH_TOKEN, project_id=PROJECT_ID, user_id=USER_ID, region_name=REGION_NAME, target_auth_url=target_auth_url, target_auth_token=target_auth_token, target_project_id=target_project_id, target_user_id=target_user_id, target_service_catalog=target_service_catalog, target_region_name=target_region, target_user_domain_name=target_user_domain_name, target_project_domain_name=target_project_domain_name, target_insecure=target_insecure)
    target_client.get(API_URL)
    self.assertTrue(m.called_once)
    headers = self.assertExpectedAuthHeaders()
    self.assertEqual(target_auth_url, headers['X-Target-Auth-Uri'])
    self.assertEqual(target_auth_token, headers['X-Target-Auth-Token'])
    self.assertEqual(target_user_id, headers['X-Target-User-Id'])
    self.assertEqual(target_project_id, headers['X-Target-Project-Id'])
    self.assertEqual(str(target_insecure), headers['X-Target-Insecure'])
    self.assertEqual(target_region, headers['X-Target-Region-Name'])
    self.assertEqual(target_user_domain_name, headers['X-Target-User-Domain-Name'])
    self.assertEqual(target_project_domain_name, headers['X-Target-Project-Domain-Name'])
    catalog = base64.b64encode(target_service_catalog.encode('utf-8'))
    self.assertEqual(catalog, headers['X-Target-Service-Catalog'])