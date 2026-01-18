from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_delete_properties_basic(self):
    self.start_server()
    path = '/v2/metadefs/namespaces'
    md_resource = self._create_metadef_resource(path, data=GLOBAL_NAMESPACE_DATA)
    namespace = md_resource['namespace']
    self.assertEqual('MyNamespace', namespace)
    self.assertIn('properties', md_resource)
    path = '/v2/metadefs/namespaces/%s/properties' % namespace
    resp = self.api_delete(path)
    self.assertEqual(204, resp.status_code)
    path = '/v2/metadefs/namespaces/%s' % namespace
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertNotIn('properties', md_resource)
    self.assertEqual('MyNamespace', namespace)
    path = '/v2/metadefs/namespaces/%s/properties' % namespace
    data = {'name': 'MyProperty', 'title': 'test_property', 'description': 'Test property for My Namespace', 'type': 'string'}
    md_resource = self._create_metadef_resource(path, data=data)
    self.assertEqual('MyProperty', md_resource['name'])
    path = '/v2/metadefs/namespaces/%s/properties' % namespace
    self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '@'})
    resp = self.api_delete(path)
    self.assertEqual(403, resp.status_code)
    path = '/v2/metadefs/namespaces/non-existing/properties'
    self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@'})
    resp = self.api_delete(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '!'})
    path = '/v2/metadefs/namespaces/%s/properties' % namespace
    resp = self.api_delete(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@'})
    self._verify_forbidden_converted_to_not_found(path, 'DELETE')