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
def test_non_aws_region(self):
    self.ec2 = boto.ec2.connect_to_region('foo', https_connection_factory=self.https_connection_factory, aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key', region=RegionInfo(name='foo', endpoint='https://foo.com/bar'))
    self.assertEqual('https://foo.com/bar', self.ec2.host)