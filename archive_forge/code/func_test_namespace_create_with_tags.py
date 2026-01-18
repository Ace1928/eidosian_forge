from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_namespace_create_with_tags(self):
    self.start_server()
    path = '/v2/metadefs/namespaces'
    data = {'tags': [{'name': 'MyTag'}]}
    data.update(NAME_SPACE1)
    md_resource = self._create_metadef_resource(path=path, data=data)
    self.assertEqual('MyNamespace', md_resource['namespace'])
    self.assertEqual('MyTag', md_resource['tags'][0]['name'])
    data.update(NAME_SPACE2)
    self.set_policy_rules({'add_metadef_tag': '!', 'get_metadef_namespace': '@'})
    resp = self.api_post(path, json=data)
    self.assertEqual(403, resp.status_code)