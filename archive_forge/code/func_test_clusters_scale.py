from saharaclient.api import clusters as cl
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_clusters_scale(self):
    url = self.URL + '/clusters/id'
    self.responses.put(url, status_code=202, json=self.body)
    scale_body = {'resize_node_groups': [{'count': 2, 'name': 'name1'}], 'add_node_groups': [{'count': 1, 'name': 'name2', 'node_group_template_id': 'id'}]}
    resp = self.client.clusters.scale('id', scale_body)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(scale_body, json.loads(self.responses.last_request.body))
    self.assertIsInstance(resp, cl.Cluster)
    self.assertFields(self.body, resp)