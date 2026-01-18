import ast
import ddt
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient import api_versions
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data(*set(['2.45', api_versions.MAX_VERSION]))
def test_delete_access_rule_metadata(self, microversion):
    self.skip_if_microversion_not_supported(microversion)
    md = {'key1': 'value1', 'key2': 'value2'}
    access = self._test_create_list_access_rule_for_share(metadata=md, microversion=microversion)
    get_access = self.user_client.access_show(access['id'], microversion=microversion)
    self.assertEqual(access['id'], get_access['id'])
    self.assertEqual(md, ast.literal_eval(get_access['metadata']))
    self.user_client.access_unset_metadata(access['id'], keys=['key1', 'key2'], microversion=microversion)
    get_access = self.user_client.access_show(access['id'], microversion=microversion)
    self.assertEqual({}, ast.literal_eval(get_access['metadata']))
    self.assertEqual(access['id'], get_access['id'])