from datetime import datetime, timedelta
from mock import MagicMock, Mock
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
import boto.ec2
from boto.regioninfo import RegionInfo
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.ec2.connection import EC2Connection
from boto.ec2.snapshot import Snapshot
from boto.ec2.reservedinstance import ReservedInstancesConfiguration
from boto.compat import http_client
def test_describe_spot_instance_requets(self):
    self.set_http_response(status_code=200)
    response = self.ec2.get_all_spot_instance_requests()
    self.assertEqual(len(response), 1)
    spotrequest = response[0]
    self.assertEqual(spotrequest.id, 'sir-id')
    self.assertEqual(spotrequest.price, 0.003)
    self.assertEqual(spotrequest.type, 'one-time')
    self.assertEqual(spotrequest.state, 'active')
    self.assertEqual(spotrequest.fault, None)
    self.assertEqual(spotrequest.valid_from, None)
    self.assertEqual(spotrequest.valid_until, None)
    self.assertEqual(spotrequest.launch_group, 'mylaunchgroup')
    self.assertEqual(spotrequest.launched_availability_zone, 'us-east-1d')
    self.assertEqual(spotrequest.product_description, 'Linux/UNIX')
    self.assertEqual(spotrequest.availability_zone_group, None)
    self.assertEqual(spotrequest.create_time, '2012-10-19T18:07:05.000Z')
    self.assertEqual(spotrequest.instance_id, 'i-id')
    launch_spec = spotrequest.launch_specification
    self.assertEqual(launch_spec.key_name, 'mykeypair')
    self.assertEqual(launch_spec.instance_type, 't1.micro')
    self.assertEqual(launch_spec.image_id, 'ami-id')
    self.assertEqual(launch_spec.placement, None)
    self.assertEqual(launch_spec.kernel, None)
    self.assertEqual(launch_spec.ramdisk, None)
    self.assertEqual(launch_spec.monitored, False)
    self.assertEqual(launch_spec.subnet_id, None)
    self.assertEqual(launch_spec.block_device_mapping, None)
    self.assertEqual(launch_spec.instance_profile, None)
    self.assertEqual(launch_spec.ebs_optimized, False)
    status = spotrequest.status
    self.assertEqual(status.code, 'fulfilled')
    self.assertEqual(status.update_time, '2012-10-19T18:09:26.000Z')
    self.assertEqual(status.message, 'Your Spot request is fulfilled.')