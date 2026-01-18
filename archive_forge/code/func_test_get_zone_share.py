import uuid
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_get_zone_share(self):
    orig_zone_share = self.operator_cloud.dns.create_zone_share(self.zone, target_project_id=self.demo_project_id)
    self.addCleanup(self.operator_cloud.dns.delete_zone_share, self.zone, orig_zone_share)
    zone_share = self.operator_cloud.dns.get_zone_share(self.zone, orig_zone_share)
    self.assertEqual(self.zone.id, zone_share.zone_id)
    self.assertEqual(self.project_id, zone_share.project_id)
    self.assertEqual(self.demo_project_id, zone_share.target_project_id)
    self.assertEqual(orig_zone_share.id, zone_share.id)
    self.assertEqual(orig_zone_share.created_at, zone_share.created_at)
    self.assertEqual(orig_zone_share.updated_at, zone_share.updated_at)