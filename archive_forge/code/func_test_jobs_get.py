from saharaclient.api import jobs
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_jobs_get(self):
    url = self.URL + '/jobs/id'
    self.responses.get(url, json={'job': self.body})
    resp = self.client.jobs.get('id')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, jobs.Job)
    self.assertFields(self.body, resp)