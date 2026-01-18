from saharaclient.api import node_group_templates as ng
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_node_group_template_list(self):
    url = self.URL + '/node-group-templates'
    self.responses.get(url, json={'node_group_templates': [self.body]})
    resp = self.client.node_group_templates.list()
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp[0], ng.NodeGroupTemplate)
    self.assertFields(self.body, resp[0])