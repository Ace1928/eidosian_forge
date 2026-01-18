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
def test_autoscaling_group_put_notification_configuration(self):
    self.set_http_response(status_code=200)
    limits = self.service_connection.get_account_limits()
    self.assert_request_parameters({'Action': 'DescribeAccountLimits'}, ignore_params_values=['Version'])
    self.assertEqual(limits.max_autoscaling_groups, 6)
    self.assertEqual(limits.max_launch_configurations, 3)