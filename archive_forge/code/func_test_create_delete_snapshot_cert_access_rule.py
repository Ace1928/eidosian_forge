from tempest.lib import exceptions as tempest_lib_exc
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_create_delete_snapshot_cert_access_rule(self):
    snapshot = self.create_snapshot(share=self.share['id'], client=self.get_user_client(), cleanup_in_class=False)
    self._create_delete_access_rule(snapshot['id'], 'cert', self.access_to['cert'][0])