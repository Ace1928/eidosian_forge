from saharaclient.api import job_executions as je
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_job_executions_get(self):
    url = self.URL + '/job-executions/id'
    self.responses.get(url, json={'job_execution': self.response})
    resp = self.client.job_executions.get('id')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, je.JobExecution)
    self.assertFields(self.response, resp)