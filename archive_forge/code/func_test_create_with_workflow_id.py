from oslo_serialization import jsonutils
from mistralclient.api import base as api_base
from mistralclient.api.v2 import executions
from mistralclient.tests.unit.v2 import base
def test_create_with_workflow_id(self):
    self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, json=EXEC, status_code=201)
    body = {'workflow_id': EXEC['workflow_id'], 'description': '', 'input': jsonutils.dumps(EXEC['input'])}
    ex = self.executions.create(EXEC['workflow_id'], workflow_input=EXEC['input'])
    self.assertIsNotNone(ex)
    self.assertDictEqual(executions.Execution(self.executions, EXEC).to_dict(), ex.to_dict())
    self.assertDictEqual(body, self.requests_mock.last_request.json())