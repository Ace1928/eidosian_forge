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
def test_autoscaling_group_vpc_zone_identifier_multi(self):
    self.set_http_response(status_code=200)
    autoscale = AutoScalingGroup(name='foo', vpc_zone_identifier='vpc_zone_1,vpc_zone_2')
    self.service_connection.create_auto_scaling_group(autoscale)
    self.assert_request_parameters({'Action': 'CreateAutoScalingGroup', 'AutoScalingGroupName': 'foo', 'VPCZoneIdentifier': 'vpc_zone_1,vpc_zone_2'}, ignore_params_values=['MaxSize', 'MinSize', 'LaunchConfigurationName', 'Version'])