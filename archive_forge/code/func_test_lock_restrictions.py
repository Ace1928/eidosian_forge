from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as lib_exc
from manilaclient.tests.functional.osc import base
from manilaclient.tests.functional import utils
def test_lock_restrictions(self):
    """A user can't update or delete a lock created by another user."""
    lock = self.create_resource_lock(self.share['id'], client=self.admin_client, add_cleanup=False)
    self.assertEqual('admin', lock['lock_context'])
    self.assertRaises(lib_exc.CommandFailed, self.openstack, f"share lock set {lock['id']} --reason 'i cannot do this'", client=self.user_client)
    self.assertRaises(lib_exc.CommandFailed, self.openstack, f'share lock unset {lock['id']} --reason', client=self.user_client)
    self.assertRaises(lib_exc.CommandFailed, self.openstack, f'share lock delete {lock['id']} ', client=self.user_client)
    self.openstack(f'share lock set --lock-reason "I can do this" {lock['id']}', client=self.admin_client)
    self.openstack(f'share lock delete {lock['id']}', client=self.admin_client)