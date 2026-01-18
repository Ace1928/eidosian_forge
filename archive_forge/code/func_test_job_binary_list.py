from saharaclient.api import job_binaries as jb
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_job_binary_list(self):
    url = self.URL + '/job-binaries'
    self.responses.get(url, json={'binaries': [self.body]})
    resp = self.client.job_binaries.list()
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp[0], jb.JobBinaries)
    self.assertFields(self.body, resp[0])