from saharaclient.api import job_types as jt
from saharaclient.tests.unit import base
def test_job_types_list(self):
    url = self.URL + '/job-types'
    self.responses.get(url, json={'job_types': [self.body]})
    resp = self.client.job_types.list()
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp[0], jt.JobType)
    self.assertFields(self.body, resp[0])