from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as lib_exc
from manilaclient.tests.functional.osc import base
from manilaclient.tests.functional import utils
def test_lock_create_show_use_delete(self):
    """Create a deletion lock on share, view it, try it and remove."""
    lock = self.create_resource_lock(self.share['id'], lock_reason='tigers rule', client=self.user_client, add_cleanup=False)
    client_user_id = self.openstack('token issue -c user_id -f value', client=self.user_client).strip()
    client_project_id = self.openstack('token issue -c project_id -f value', client=self.user_client).strip()
    self.assertEqual(self.share['id'], lock['resource_id'])
    self.assertEqual('delete', lock['resource_action'])
    self.assertEqual(client_user_id, lock['user_id'])
    self.assertEqual(client_project_id, lock['project_id'])
    self.assertEqual('user', lock['lock_context'])
    self.assertEqual('tigers rule', lock['lock_reason'])
    lock_show = self.dict_result('share', f'lock show {lock['id']}')
    self.assertEqual(lock['id'], lock_show['ID'])
    self.assertEqual(lock['lock_context'], lock_show['Lock Context'])
    self.assertRaises(lib_exc.CommandFailed, self.openstack, f'share delete {self.share['id']}')
    self.openstack(f'share lock delete {lock['id']}', client=self.user_client)
    self.assertRaises(lib_exc.CommandFailed, self.openstack, f'share lock show {lock['id']}')