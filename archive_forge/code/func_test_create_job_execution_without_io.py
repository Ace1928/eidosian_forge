from saharaclient.api import job_executions as je
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_create_job_execution_without_io(self):
    url = self.URL + '/jobs/job_id/execute'
    self.responses.post(url, status_code=202, json={'job_execution': self.response})
    resp = self.client.job_executions.create(**self.body)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(self.response, json.loads(self.responses.last_request.body))
    self.assertIsInstance(resp, je.JobExecution)
    self.assertFields(self.response, resp)