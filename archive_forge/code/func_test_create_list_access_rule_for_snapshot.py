from tempest.lib import exceptions as tempest_lib_exc
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_create_list_access_rule_for_snapshot(self):
    snapshot = self.create_snapshot(share=self.share['id'], client=self.get_user_client(), cleanup_in_class=False)
    access = self._test_create_list_access_rule_for_snapshot(snapshot['id'])
    access_list = self.user_client.list_access(snapshot['id'], is_snapshot=True)
    for i in range(5):
        self.assertIn(access[i]['id'], [access_list[j]['id'] for j in range(5)])
        self.assertIn(access[i]['access_type'], [access_list[j]['access_type'] for j in range(5)])
        self.assertIn(access[i]['access_to'], [access_list[j]['access_to'] for j in range(5)])
        self.assertIsNotNone(access_list[i]['access_type'])
        self.assertIsNotNone(access_list[i]['access_to'])