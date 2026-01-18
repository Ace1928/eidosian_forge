import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
def test_all_task_api(self):
    path = '/v2/tasks'
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    content_dict = json.loads(content)
    self.assertEqual(http.client.OK, response.status)
    self.assertFalse(content_dict['tasks'])
    task_id = 'NON_EXISTENT_TASK'
    path = '/v2/tasks/%s' % task_id
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.NOT_FOUND, response.status)
    task_owner = 'tenant1'
    data, req_input = self._post_new_task(owner=task_owner)
    task_id = data['id']
    path = '/v2/tasks/%s' % task_id
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    self._wait_on_task_execution(max_wait=10)
    path = '/v2/tasks'
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    self.assertIsNotNone(content)
    data = json.loads(content)
    self.assertIsNotNone(data)
    self.assertEqual(1, len(data['tasks']))
    expected_keys = set(['id', 'expires_at', 'type', 'owner', 'status', 'created_at', 'updated_at', 'self', 'schema'])
    task = data['tasks'][0]
    self.assertEqual(expected_keys, set(task.keys()))
    self.assertEqual(req_input['type'], task['type'])
    self.assertEqual(task_owner, task['owner'])
    self.assertEqual('success', task['status'])
    self.assertIsNotNone(task['created_at'])
    self.assertIsNotNone(task['updated_at'])