from saharaclient.api import job_executions as je
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_job_executions_list(self):
    url = self.URL + '/job-executions'
    self.responses.get(url, json={'job_executions': [self.response]})
    resp = self.client.job_executions.list()
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp[0], je.JobExecution)
    self.assertFields(self.response, resp[0])