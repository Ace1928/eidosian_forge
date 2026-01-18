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
def test_describe_vpc_attribute(self):
    self.set_http_response(status_code=200)
    parsed = self.ec2.describe_vpc_attribute('vpc-id', 'enableDnsHostnames')
    self.assertEqual(parsed.vpc_id, 'vpc-id')
    self.assertFalse(parsed.enable_dns_hostnames)
    self.assert_request_parameters({'Action': 'DescribeVpcAttribute', 'VpcId': 'vpc-id', 'Attribute': 'enableDnsHostnames'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])