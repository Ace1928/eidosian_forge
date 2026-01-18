from tempest.lib import exceptions as tempest_lib_exc
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_create_list_access_rule_for_snapshot_select_column(self):
    snapshot = self.create_snapshot(share=self.share['id'], client=self.get_user_client(), cleanup_in_class=False)
    self._test_create_list_access_rule_for_snapshot(snapshot['id'])
    access_list = self.user_client.list_access(snapshot['id'], columns='access_type,access_to', is_snapshot=True)
    self.assertTrue(any((x['Access_Type'] is not None for x in access_list)))
    self.assertTrue(any((x['Access_To'] is not None for x in access_list)))