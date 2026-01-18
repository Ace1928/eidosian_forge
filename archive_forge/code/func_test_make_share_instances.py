from unittest import mock
from keystoneauth1 import adapter
from openstack.shared_file_system.v2 import share_instance
from openstack.tests.unit import base
def test_make_share_instances(self):
    share_instance_resource = share_instance.ShareInstance(**EXAMPLE)
    self.assertEqual(EXAMPLE['status'], share_instance_resource.status)
    self.assertEqual(EXAMPLE['progress'], share_instance_resource.progress)
    self.assertEqual(EXAMPLE['share_id'], share_instance_resource.share_id)
    self.assertEqual(EXAMPLE['availability_zone'], share_instance_resource.availability_zone)
    self.assertEqual(EXAMPLE['replica_state'], share_instance_resource.replica_state)
    self.assertEqual(EXAMPLE['created_at'], share_instance_resource.created_at)
    self.assertEqual(EXAMPLE['cast_rules_to_readonly'], share_instance_resource.cast_rules_to_readonly)
    self.assertEqual(EXAMPLE['share_network_id'], share_instance_resource.share_network_id)
    self.assertEqual(EXAMPLE['share_server_id'], share_instance_resource.share_server_id)
    self.assertEqual(EXAMPLE['host'], share_instance_resource.host)
    self.assertEqual(EXAMPLE['access_rules_status'], share_instance_resource.access_rules_status)
    self.assertEqual(EXAMPLE['id'], share_instance_resource.id)