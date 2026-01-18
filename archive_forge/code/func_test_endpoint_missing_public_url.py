import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_endpoint_missing_public_url(self):
    env_headers = {'HTTP_X_SERVICE_CATALOG': '[{"endpoints_links": [],\n                            "endpoints": [{"adminURL":\n                                           "http://admin_host:8774",\n                                           "region": "RegionOne",\n                                           "internalURL":\n                                           "http://internal_host:8774"}],\n                             "type": "compute",\n                             "name": "nova"}]', 'HTTP_X_USER_ID': 'user_id', 'HTTP_X_USER_NAME': 'user_name', 'HTTP_X_AUTH_TOKEN': 'token', 'HTTP_X_PROJECT_ID': 'tenant_id', 'HTTP_X_IDENTITY_STATUS': 'Confirmed', 'REQUEST_METHOD': 'GET'}
    url = 'http://admin_host:8774/v2/' + str(uuid.uuid4()) + '/servers'
    payload = self.get_payload('GET', url, environ=env_headers)
    self.assertEqual(payload['target']['addresses'][2]['url'], 'unknown')