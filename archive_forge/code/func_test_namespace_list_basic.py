from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_list_basic(self):
    self.start_server()
    path = '/v2/metadefs/namespaces'
    md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE1)
    self.assertEqual('MyNamespace', md_resource['namespace'])
    path = '/v2/metadefs/namespaces'
    NAME_SPACE2['visibility'] = 'public'
    md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE2)
    self.assertEqual('MySecondNamespace', md_resource['namespace'])
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertEqual(2, len(md_resource['namespaces']))
    self.set_policy_rules({'get_metadef_namespaces': '!', 'get_metadef_namespace': '@'})
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)