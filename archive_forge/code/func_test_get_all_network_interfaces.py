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
def test_get_all_network_interfaces(self):
    self.set_http_response(status_code=200)
    result = self.ec2.get_all_network_interfaces(network_interface_ids=['eni-0f62d866'])
    self.assert_request_parameters({'Action': 'DescribeNetworkInterfaces', 'NetworkInterfaceId.1': 'eni-0f62d866'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0].id, 'eni-0f62d866')