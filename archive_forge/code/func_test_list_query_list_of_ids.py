from openstack.network.v2 import security_group
from openstack.tests.functional import base
def test_list_query_list_of_ids(self):
    ids = [o.id for o in self.user_cloud.network.security_groups(id=[self.ID])]
    self.assertIn(self.ID, ids)