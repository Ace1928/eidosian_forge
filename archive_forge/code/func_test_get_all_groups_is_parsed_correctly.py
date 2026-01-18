import base64
from datetime import datetime
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.autoscale import AutoScaleConnection
from boto.ec2.autoscale.group import AutoScalingGroup
from boto.ec2.autoscale.policy import ScalingPolicy
from boto.ec2.autoscale.tag import Tag
from boto.ec2.blockdevicemapping import EBSBlockDeviceType, BlockDeviceMapping
from boto.ec2.autoscale import launchconfig, LaunchConfiguration
def test_get_all_groups_is_parsed_correctly(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_all_groups(names=['test_group'])
    self.assertEqual(len(response), 1, response)
    as_group = response[0]
    self.assertEqual(as_group.availability_zones, ['us-east-1c', 'us-east-1a'])
    self.assertEqual(as_group.default_cooldown, 300)
    self.assertEqual(as_group.desired_capacity, 1)
    self.assertEqual(as_group.enabled_metrics, [])
    self.assertEqual(as_group.health_check_period, 0)
    self.assertEqual(as_group.health_check_type, 'EC2')
    self.assertEqual(as_group.launch_config_name, 'test_launchconfig')
    self.assertEqual(as_group.load_balancers, [])
    self.assertEqual(as_group.min_size, 1)
    self.assertEqual(as_group.max_size, 2)
    self.assertEqual(as_group.name, 'test_group')
    self.assertEqual(as_group.suspended_processes, [])
    self.assertEqual(as_group.tags, [])
    self.assertEqual(as_group.termination_policies, ['OldestInstance', 'OldestLaunchConfiguration'])
    self.assertEqual(as_group.instance_id, 'Something')