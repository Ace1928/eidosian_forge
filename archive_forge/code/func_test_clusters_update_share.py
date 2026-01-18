from saharaclient.api import clusters as cl
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_clusters_update_share(self):
    url = self.URL + '/clusters/id'
    update_body = {'name': 'new_name', 'description': 'descr', 'shares': self.test_shares}
    self.responses.patch(url, status_code=202, json=update_body)
    resp = self.client.clusters.update('id', name='new_name', description='descr', shares=self.test_shares)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, cl.Cluster)
    self.assertEqual(update_body, json.loads(self.responses.last_request.body))