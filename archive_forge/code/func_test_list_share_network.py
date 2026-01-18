from openstack.shared_file_system.v2 import share_network as _share_network
from openstack.tests.functional.shared_file_system import base
def test_list_share_network(self):
    share_nets = self.user_cloud.shared_file_system.share_networks(details=False)
    self.assertGreater(len(list(share_nets)), 0)
    for share_net in share_nets:
        for attribute in ('id', 'name', 'created_at', 'updated_at'):
            self.assertTrue(hasattr(share_net, attribute))