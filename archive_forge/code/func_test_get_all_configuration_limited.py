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
def test_get_all_configuration_limited(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_all_launch_configurations(max_records=10, names=['my-test1', 'my-test2'])
    self.assert_request_parameters({'Action': 'DescribeLaunchConfigurations', 'MaxRecords': 10, 'LaunchConfigurationNames.member.1': 'my-test1', 'LaunchConfigurationNames.member.2': 'my-test2'}, ignore_params_values=['Version'])