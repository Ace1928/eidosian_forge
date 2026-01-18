from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as lib_exc
from manilaclient.tests.functional.osc import base
from manilaclient.tests.functional import utils
def test_lock_set_unset_lock_reason(self):
    lock = self.create_resource_lock(self.share['id'], client=self.user_client)
    self.assertEqual('None', lock['lock_reason'])
    self.openstack(f"share lock set --lock-reason 'updated reason' {lock['id']}")
    lock_show = self.dict_result('share', f'lock show {lock['id']}')
    self.assertEqual('updated reason', lock_show['Lock Reason'])
    self.openstack(f'share lock unset --lock-reason {lock['id']}')
    lock_show = self.dict_result('share', f'lock show {lock['id']}')
    self.assertEqual('None', lock_show['Lock Reason'])