import ast
import ddt
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient import api_versions
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data('1.0', '2.0', '2.6', '2.7')
def test_create_list_access_rule_for_share_select_column(self, microversion):
    self.skip_if_microversion_not_supported(microversion)
    self._test_create_list_access_rule_for_share(microversion=microversion)
    access_list = self.user_client.list_access(self.share['id'], columns='access_type,access_to', microversion=microversion)
    self.assertTrue(any((a['Access_Type'] is not None for a in access_list)))
    self.assertTrue(any((a['Access_To'] is not None for a in access_list)))
    self.assertTrue(all(('Access_Level' not in a for a in access_list)))
    self.assertTrue(all(('access_level' not in a for a in access_list)))