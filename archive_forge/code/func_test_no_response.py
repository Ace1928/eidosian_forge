from unittest import mock
import uuid
import fixtures
import webob
from keystonemiddleware.tests.unit.audit import base
def test_no_response(self):
    middleware = self.create_simple_middleware()
    url = 'http://admin_host:8774/v2/' + str(uuid.uuid4()) + '/servers'
    req = webob.Request.blank(url, environ=self.get_environ_header('GET'), remote_addr='192.168.0.1')
    req.environ['audit.context'] = {}
    middleware._process_request(req)
    payload = req.environ['cadf_event'].as_dict()
    middleware._process_response(req, None)
    payload2 = req.environ['cadf_event'].as_dict()
    self.assertEqual(payload['id'], payload2['id'])
    self.assertEqual(payload['tags'], payload2['tags'])
    self.assertEqual(payload2['outcome'], 'unknown')
    self.assertNotIn('reason', payload2)
    self.assertEqual(len(payload2['reporterchain']), 1)
    self.assertEqual(payload2['reporterchain'][0]['role'], 'modifier')
    self.assertEqual(payload2['reporterchain'][0]['reporter']['id'], 'target')