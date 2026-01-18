import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_get_list(self):
    path = '/v2/' + str(uuid.uuid4()) + '/servers'
    url = 'http://admin_host:8774' + path
    payload = self.get_payload('GET', url)
    self.assertEqual(payload['action'], 'read/list')
    self.assertEqual(payload['typeURI'], 'http://schemas.dmtf.org/cloud/audit/1.0/event')
    self.assertEqual(payload['outcome'], 'pending')
    self.assertEqual(payload['eventType'], 'activity')
    self.assertEqual(payload['target']['name'], 'nova')
    self.assertEqual(payload['target']['id'], 'resource_id')
    self.assertEqual(payload['target']['typeURI'], 'service/compute/servers')
    self.assertEqual(len(payload['target']['addresses']), 3)
    self.assertEqual(payload['target']['addresses'][0]['name'], 'admin')
    self.assertEqual(payload['target']['addresses'][0]['url'], 'http://admin_host:8774')
    self.assertEqual(payload['initiator']['id'], 'user_id')
    self.assertEqual(payload['initiator']['name'], 'user_name')
    self.assertEqual(payload['initiator']['project_id'], 'tenant_id')
    self.assertEqual(payload['initiator']['host']['address'], '192.168.0.1')
    self.assertEqual(payload['initiator']['typeURI'], 'service/security/account/user')
    self.assertNotEqual(payload['initiator']['credential']['token'], 'token')
    self.assertEqual(payload['initiator']['credential']['identity_status'], 'Confirmed')
    self.assertNotIn('reason', payload)
    self.assertNotIn('reporterchain', payload)
    self.assertEqual(payload['observer']['id'], 'target')
    self.assertEqual(path, payload['requestPath'])