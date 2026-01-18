from saharaclient.api import clusters as cl
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_create_cluster_with_template(self):
    url = self.URL + '/clusters'
    self.responses.post(url, status_code=202, json={'cluster': self.body})
    resp = self.client.clusters.create(**self.body)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(self.body, json.loads(self.responses.last_request.body))
    self.assertIsInstance(resp, cl.Cluster)
    self.assertFields(self.body, resp)