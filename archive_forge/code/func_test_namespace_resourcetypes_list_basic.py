from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_resourcetypes_list_basic(self):
    self.start_server()
    self.load_data(create_resourcetypes=True)
    namespace = NAME_SPACE1['namespace']
    path = '/v2/metadefs/namespaces/%s/resource_types' % namespace
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertEqual(2, len(md_resource['resource_type_associations']))
    self.set_policy_rules({'list_metadef_resource_types': '!', 'get_metadef_namespace': '@'})
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'list_metadef_resource_types': '!', 'get_metadef_namespace': '!'})
    resp = self.api_get(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'list_metadef_resource_types': '@', 'get_metadef_resource_type': '!', 'get_metadef_namespace': '@'})
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertEqual(0, len(md_resource['resource_type_associations']))
    self.set_policy_rules({'list_metadef_resource_types': '@', 'get_metadef_resource_type': '@', 'get_metadef_namespace': '@'})
    self._verify_forbidden_converted_to_not_found(path, 'GET')