from saharaclient.api import job_executions as je
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_job_executions_delete(self):
    url = self.URL + '/job-executions/id'
    self.responses.delete(url, status_code=204)
    self.client.job_executions.delete('id')
    self.assertEqual(url, self.responses.last_request.url)