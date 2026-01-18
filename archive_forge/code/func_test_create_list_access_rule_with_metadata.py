import ast
import ddt
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient import api_versions
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data(*set(['2.45', api_versions.MAX_VERSION]))
def test_create_list_access_rule_with_metadata(self, microversion):
    self.skip_if_microversion_not_supported(microversion)
    md1 = {'key1': 'value1', 'key2': 'value2'}
    md2 = {'key3': 'value3', 'key4': 'value4'}
    self._test_create_list_access_rule_for_share(metadata=md1, microversion=microversion)
    access = self._test_create_list_access_rule_for_share(metadata=md2, microversion=microversion)
    access_list = self.user_client.list_access(self.share['id'], metadata={'key4': 'value4'}, microversion=microversion)
    self.assertEqual(1, len(access_list))
    get_access = self.user_client.access_show(access_list[0]['id'], microversion=microversion)
    metadata = ast.literal_eval(get_access['metadata'])
    self.assertEqual(2, len(metadata))
    self.assertIn('key3', metadata)
    self.assertIn('key4', metadata)
    self.assertEqual(md2['key3'], metadata['key3'])
    self.assertEqual(md2['key4'], metadata['key4'])
    self.assertEqual(access['id'], access_list[0]['id'])
    self.user_client.access_deny(access['share_id'], access['id'])
    self.user_client.wait_for_access_rule_deletion(access['share_id'], access['id'])