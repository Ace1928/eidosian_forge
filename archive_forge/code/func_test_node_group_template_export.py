from saharaclient.api import node_group_templates as ng
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_node_group_template_export(self):
    url = self.URL + '/node-group-templates/id/export'
    self.responses.get(url, json={'node_group_template': self.body})
    resp = self.client.node_group_templates.export('id')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, ng.NodeGroupTemplate)
    self.assertDictsEqual(self.body, resp.__dict__[u'node_group_template'])