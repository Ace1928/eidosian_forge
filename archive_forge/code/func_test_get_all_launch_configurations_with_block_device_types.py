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
def test_get_all_launch_configurations_with_block_device_types(self):
    self.set_http_response(status_code=200)
    self.service_connection.use_block_device_types = True
    response = self.service_connection.get_all_launch_configurations()
    self.assertTrue(isinstance(response, list))
    self.assertEqual(len(response), 1)
    self.assertTrue(isinstance(response[0], LaunchConfiguration))
    self.assertEqual(response[0].associate_public_ip_address, True)
    self.assertEqual(response[0].name, 'my-test-lc')
    self.assertEqual(response[0].instance_type, 'm1.small')
    self.assertEqual(response[0].launch_configuration_arn, 'arn:aws:autoscaling:us-east-1:803981987763:launchConfiguration:9dbbbf87-6141-428a-a409-0752edbe6cad:launchConfigurationName/my-test-lc')
    self.assertEqual(response[0].image_id, 'ami-514ac838')
    self.assertTrue(isinstance(response[0].instance_monitoring, launchconfig.InstanceMonitoring))
    self.assertEqual(response[0].instance_monitoring.enabled, 'true')
    self.assertEqual(response[0].ebs_optimized, False)
    self.assertEqual(response[0].block_device_mappings['/dev/xvdb'].ephemeral_name, 'ephemeral0')
    self.assertEqual(response[0].block_device_mappings['/dev/xvdc'].ephemeral_name, 'ephemeral1')
    self.assertEqual(response[0].block_device_mappings['/dev/xvdp'].snapshot_id, 'snap-1234abcd')
    self.assertEqual(response[0].block_device_mappings['/dev/xvdp'].delete_on_termination, True)
    self.assertEqual(response[0].block_device_mappings['/dev/xvdp'].iops, 1000)
    self.assertEqual(response[0].block_device_mappings['/dev/xvdp'].size, 100)
    self.assertEqual(response[0].block_device_mappings['/dev/xvdp'].volume_type, 'io1')
    self.assertEqual(response[0].block_device_mappings['/dev/xvdh'].delete_on_termination, False)
    self.assertEqual(response[0].block_device_mappings['/dev/xvdh'].iops, 2000)
    self.assertEqual(response[0].block_device_mappings['/dev/xvdh'].size, 200)
    self.assertEqual(response[0].block_device_mappings['/dev/xvdh'].volume_type, 'io1')
    self.assert_request_parameters({'Action': 'DescribeLaunchConfigurations'}, ignore_params_values=['Version'])