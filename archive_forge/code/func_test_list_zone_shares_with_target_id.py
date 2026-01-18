import uuid
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_list_zone_shares_with_target_id(self):
    zone_share = self.operator_cloud.dns.create_zone_share(self.zone, target_project_id=self.demo_project_id)
    self.addCleanup(self.operator_cloud.dns.delete_zone_share, self.zone, zone_share)
    target_ids = [o.target_project_id for o in self.operator_cloud.dns.zone_shares(self.zone, target_project_id=self.demo_project_id)]
    self.assertIn(self.demo_project_id, target_ids)