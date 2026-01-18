from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_create_basic(self):
    self.start_server()
    path = '/v2/metadefs/namespaces'
    md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE1)
    self.assertEqual('MyNamespace', md_resource['namespace'])
    self.set_policy_rules({'add_metadef_namespace': '!', 'get_metadef_namespace': '@'})
    resp = self.api_post(path, json=NAME_SPACE2)
    self.assertEqual(403, resp.status_code)