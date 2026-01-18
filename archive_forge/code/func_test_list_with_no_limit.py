from mistralclient.api.v2 import action_executions
from mistralclient.tests.unit.v2 import base
def test_list_with_no_limit(self):
    self.requests_mock.get(self.TEST_URL + URL_TEMPLATE, json={'action_executions': [ACTION_EXEC]})
    action_execution_list = self.action_executions.list(limit=-1)
    self.assertEqual(1, len(action_execution_list))
    last_request = self.requests_mock.last_request
    self.assertNotIn('limit', last_request.qs)