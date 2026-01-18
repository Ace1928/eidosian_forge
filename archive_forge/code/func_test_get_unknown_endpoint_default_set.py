import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_get_unknown_endpoint_default_set(self):
    with open(self.audit_map, 'w') as f:
        f.write('[DEFAULT]\n')
        f.write('target_endpoint_type = compute\n')
        f.write('[path_keywords]\n')
        f.write('servers = server\n\n')
        f.write('[service_endpoints]\n')
        f.write('compute = service/compute')
    url = 'http://unknown:8774/v2/' + str(uuid.uuid4()) + '/servers'
    payload = self.get_payload('GET', url)
    self.assertEqual(payload['action'], 'read/list')
    self.assertEqual(payload['outcome'], 'pending')
    self.assertEqual(payload['target']['name'], 'nova')
    self.assertEqual(payload['target']['id'], 'resource_id')
    self.assertEqual(payload['target']['typeURI'], 'service/compute/servers')