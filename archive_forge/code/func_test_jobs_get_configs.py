from saharaclient.api import jobs
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_jobs_get_configs(self):
    url = self.URL + '/jobs/config-hints/Pig'
    response = {'job_config': {'args': [], 'configs': []}, 'interface': []}
    self.responses.get(url, json=response)
    resp = self.client.jobs.get_configs('Pig')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, jobs.Job)
    self.assertFields(response, resp)