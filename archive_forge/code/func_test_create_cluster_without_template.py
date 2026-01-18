from saharaclient.api import clusters as cl
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_create_cluster_without_template(self):
    body = self.body.copy()
    del body['cluster_template_id']
    body.update({'default_image_id': 'image_id', 'cluster_configs': {}, 'node_groups': ['ng1', 'ng2']})
    url = self.URL + '/clusters'
    self.responses.post(url, status_code=202, json={'cluster': body})
    resp = self.client.clusters.create(**body)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(body, json.loads(self.responses.last_request.body))
    self.assertIsInstance(resp, cl.Cluster)
    self.assertFields(body, resp)