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
def test_get_classic_link_instances(self):
    self.set_http_response(status_code=200)
    response = self.ec2.get_all_classic_link_instances()
    self.assertEqual(len(response), 1)
    instance = response[0]
    self.assertEqual(instance.id, 'i-31489bd8')
    self.assertEqual(instance.vpc_id, 'vpc-9d24f8f8')
    self.assertEqual(len(instance.groups), 1)
    self.assertEqual(instance.groups[0].id, 'sg-9b4343fe')
    self.assertEqual(instance.tags, {'Name': 'hello'})