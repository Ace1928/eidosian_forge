import time
from boto.ec2.autoscale import AutoScaleConnection
from boto.ec2.autoscale.activity import Activity
from boto.ec2.autoscale.group import AutoScalingGroup, ProcessType
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.policy import AdjustmentType, MetricCollectionTypes, ScalingPolicy
from boto.ec2.autoscale.scheduled import ScheduledUpdateGroupAction
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.tag import Tag
from tests.compat import unittest
def test_ebs_optimized_regression(self):
    c = AutoScaleConnection()
    time_string = '%d' % int(time.time())
    lc_name = 'lc-%s' % time_string
    lc = LaunchConfiguration(name=lc_name, image_id='ami-2272864b', instance_type='t1.micro', ebs_optimized=True)
    c.create_launch_configuration(lc)
    self.addCleanup(c.delete_launch_configuration, lc_name)