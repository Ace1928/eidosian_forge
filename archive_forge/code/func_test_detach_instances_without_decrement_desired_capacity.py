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
def test_detach_instances_without_decrement_desired_capacity(self):
    self.set_http_response(status_code=200)
    self.service_connection.detach_instances('autoscale', ['inst2', 'inst1', 'inst4'], False)
    self.assert_request_parameters({'Action': 'DetachInstances', 'AutoScalingGroupName': 'autoscale', 'InstanceIds.member.1': 'inst2', 'InstanceIds.member.2': 'inst1', 'InstanceIds.member.3': 'inst4', 'ShouldDecrementDesiredCapacity': 'false'}, ignore_params_values=['Version'])