import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_endpoint_no_service_port(self):
    with open(self.audit_map, 'w') as f:
        f.write('[DEFAULT]\n')
        f.write('target_endpoint_type = load-balancer\n')
        f.write('[path_keywords]\n')
        f.write('loadbalancers = loadbalancer\n\n')
        f.write('[service_endpoints]\n')
        f.write('load-balancer = service/load-balancer')
    env_headers = {'HTTP_X_SERVICE_CATALOG': '[{"endpoints_links": [],\n                            "endpoints": [{"adminURL":\n                                           "http://admin_host/compute",\n                                           "region": "RegionOne",\n                                           "publicURL":\n                                           "http://public_host/compute"}],\n                             "type": "compute",\n                             "name": "nova"},\n                           {"endpoints_links": [],\n                            "endpoints": [{"adminURL":\n                                           "http://admin_host/load-balancer",\n                                           "region": "RegionOne",\n                                           "publicURL":\n                                           "http://public_host/load-balancer"}],\n                             "type": "load-balancer",\n                             "name": "octavia"}]', 'HTTP_X_USER_ID': 'user_id', 'HTTP_X_USER_NAME': 'user_name', 'HTTP_X_AUTH_TOKEN': 'token', 'HTTP_X_PROJECT_ID': 'tenant_id', 'HTTP_X_IDENTITY_STATUS': 'Confirmed', 'REQUEST_METHOD': 'GET'}
    url = 'http://admin_host/load-balancer/v2/loadbalancers/' + str(uuid.uuid4())
    payload = self.get_payload('GET', url, environ=env_headers)
    self.assertEqual(payload['target']['id'], 'octavia')