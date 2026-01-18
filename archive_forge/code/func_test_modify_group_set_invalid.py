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
def test_modify_group_set_invalid(self):
    self.set_http_response(status_code=200)
    with self.assertRaisesRegexp(TypeError, 'iterable'):
        self.ec2.modify_network_interface_attribute('id', 'groupSet', False)