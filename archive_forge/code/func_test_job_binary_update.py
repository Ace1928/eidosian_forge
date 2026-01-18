from saharaclient.api import job_binaries as jb
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_job_binary_update(self):
    url = self.URL + '/job-binaries/id'
    self.responses.put(url, status_code=202, json={'job_binary': self.update_body})
    resp = self.client.job_binaries.update('id', self.update_body)
    self.assertEqual(self.update_body['name'], resp.name)