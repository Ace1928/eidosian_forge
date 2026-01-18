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
def test_associate_address_object(self):
    self.set_http_response(status_code=200)
    result = self.ec2.associate_address_object(instance_id='i-1234', public_ip='192.0.2.1')
    self.assertEqual('eipassoc-fc5ca095', result.association_id)