from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_delete_basic(self):

    def _create_private_namespace(fn_call, data):
        path = '/v2/metadefs/namespaces'
        return fn_call(path=path, data=data)
    self.start_server()
    md_resource = _create_private_namespace(self._create_metadef_resource, NAME_SPACE1)
    self.assertEqual('MyNamespace', md_resource['namespace'])
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    resp = self.api_delete(path)
    self.assertEqual(204, resp.status_code)
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    resp = self.api_get(path)
    self.assertEqual(404, resp.status_code)
    md_resource = _create_private_namespace(self._create_metadef_resource, NAME_SPACE2)
    self.assertEqual('MySecondNamespace', md_resource['namespace'])
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '@'})
    resp = self.api_delete(path)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@'})
    path = '/v2/metadefs/namespaces/non-existing'
    resp = self.api_delete(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '!'})
    path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
    resp = self.api_delete(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@'})
    self._verify_forbidden_converted_to_not_found(path, 'DELETE')