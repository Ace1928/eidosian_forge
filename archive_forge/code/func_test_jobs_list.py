from saharaclient.api import jobs
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_jobs_list(self):
    url = self.URL + '/jobs'
    self.responses.get(url, json={'jobs': [self.body]})
    resp = self.client.jobs.list()
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp[0], jobs.Job)
    self.assertFields(self.body, resp[0])