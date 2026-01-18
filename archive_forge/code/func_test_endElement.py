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
def test_endElement(self):
    for i in [('Key', 'mykey', 'key'), ('Value', 'myvalue', 'value'), ('ResourceType', 'auto-scaling-group', 'resource_type'), ('ResourceId', 'sg-01234567', 'resource_id'), ('PropagateAtLaunch', 'true', 'propagate_at_launch')]:
        self.check_tag_attributes_set(i[0], i[1], i[2])