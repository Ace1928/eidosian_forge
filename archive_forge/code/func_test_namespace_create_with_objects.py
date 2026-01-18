from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_create_with_objects(self):
    self.start_server()
    path = '/v2/metadefs/namespaces'
    data = {'objects': [{'name': 'MyObject', 'description': 'My object for My namespace', 'properties': {'test_property': {'title': 'test_property', 'description': 'Test property for My object', 'type': 'string'}}}]}
    data.update(NAME_SPACE1)
    md_resource = self._create_metadef_resource(path=path, data=data)
    self.assertEqual('MyNamespace', md_resource['namespace'])
    self.assertEqual('MyObject', md_resource['objects'][0]['name'])
    self.set_policy_rules({'add_metadef_object': '!', 'get_metadef_namespace': '@'})
    data.update(NAME_SPACE2)
    resp = self.api_post(path, json=data)
    self.assertEqual(403, resp.status_code)