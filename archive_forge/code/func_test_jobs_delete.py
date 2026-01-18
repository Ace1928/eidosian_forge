from saharaclient.api import jobs
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_jobs_delete(self):
    url = self.URL + '/jobs/id'
    self.responses.delete(url, status_code=204)
    self.client.jobs.delete('id')
    self.assertEqual(url, self.responses.last_request.url)