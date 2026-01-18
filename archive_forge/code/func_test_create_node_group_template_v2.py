from saharaclient.api import node_group_templates as ng
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_create_node_group_template_v2(self):
    url = self.URL + '/node-group-templates'
    self.responses.post(url, status_code=202, json={'node_group_template': self.body})
    resp = self.client_v2.node_group_templates.create(**self.body)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(self.body, json.loads(self.responses.last_request.body))
    self.assertIsInstance(resp, ng.NodeGroupTemplate)
    self.assertFields(self.body, resp)