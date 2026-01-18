from oslo_serialization import jsonutils as json
from saharaclient.api import job_binary_internals as jbi
from saharaclient.tests.unit import base
def test_job_binary_internal_list(self):
    url = self.URL + '/job-binary-internals'
    self.responses.get(url, json={'binaries': [self.body]})
    resp = self.client.job_binary_internals.list()
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp[0], jbi.JobBinaryInternal)
    self.assertFields(self.body, resp[0])