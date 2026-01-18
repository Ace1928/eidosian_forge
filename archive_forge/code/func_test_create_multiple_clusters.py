from saharaclient.api import clusters as cl
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_create_multiple_clusters(self):
    url = self.URL + '/clusters/multiple'
    self.responses.post(url, status_code=202, json={'clusters': ['id1', 'id2']})
    resp = self.client.clusters.create(**self.body_with_count)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(self.body_with_count, json.loads(self.responses.last_request.body))
    self.assertIsInstance(resp, cl.Cluster)
    self.assertFields({'clusters': ['id1', 'id2']}, resp)