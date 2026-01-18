from openstack import exceptions
from openstack.shared_file_system.v2 import share as _share
from openstack.tests.functional.shared_file_system import base
def test_list_share(self):
    shares = self.user_cloud.share.shares(details=False)
    self.assertGreater(len(list(shares)), 0)
    for share in shares:
        for attribute in ('id', 'name', 'created_at', 'updated_at'):
            self.assertTrue(hasattr(share, attribute))