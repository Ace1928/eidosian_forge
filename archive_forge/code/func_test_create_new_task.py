import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
def test_create_new_task(self):
    task_data = _new_task_fixture()
    task_owner = 'tenant1'
    body_content = json.dumps(task_data)
    path = '/v2/tasks'
    response, content = self.http.request(path, 'POST', headers=minimal_task_headers(task_owner), body=body_content)
    self.assertEqual(http.client.CREATED, response.status)
    data = json.loads(content)
    task_id = data['id']
    self.assertIsNotNone(task_id)
    self.assertEqual(task_owner, data['owner'])
    self.assertEqual(task_data['type'], data['type'])
    self.assertEqual(task_data['input'], data['input'])
    task_data = _new_task_fixture(type='invalid')
    task_owner = 'tenant1'
    body_content = json.dumps(task_data)
    path = '/v2/tasks'
    response, content = self.http.request(path, 'POST', headers=minimal_task_headers(task_owner), body=body_content)
    self.assertEqual(http.client.BAD_REQUEST, response.status)
    task_data = _new_task_fixture(task_input='{something: invalid}')
    task_owner = 'tenant1'
    body_content = json.dumps(task_data)
    path = '/v2/tasks'
    response, content = self.http.request(path, 'POST', headers=minimal_task_headers(task_owner), body=body_content)
    self.assertEqual(http.client.BAD_REQUEST, response.status)
    self._wait_on_task_execution()