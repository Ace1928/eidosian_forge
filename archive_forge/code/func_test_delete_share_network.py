from openstack.shared_file_system.v2 import share_network as _share_network
from openstack.tests.functional.shared_file_system import base
def test_delete_share_network(self):
    sot = self.user_cloud.shared_file_system.delete_share_network(self.SHARE_NETWORK_ID)
    self.assertIsNone(sot)