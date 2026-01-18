import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
def test_delete_task(self):
    task_data = _new_task_fixture()
    task_owner = 'tenant1'
    body_content = json.dumps(task_data)
    path = '/v2/tasks'
    response, content = self.http.request(path, 'POST', headers=minimal_task_headers(task_owner), body=body_content)
    self.assertEqual(http.client.CREATED, response.status)
    data = json.loads(content)
    task_id = data['id']
    path = '/v2/tasks/%s' % task_id
    response, content = self.http.request(path, 'DELETE', headers=minimal_task_headers())
    self.assertEqual(http.client.METHOD_NOT_ALLOWED, response.status)
    self.assertEqual('GET', response.webob_resp.headers.get('Allow'))
    self.assertEqual(('GET',), response.webob_resp.allow)
    self.assertEqual(('GET',), response.allow)
    path = '/v2/tasks/%s' % task_id
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    self.assertIsNotNone(content)
    self._wait_on_task_execution()