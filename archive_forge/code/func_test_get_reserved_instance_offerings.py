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
def test_get_reserved_instance_offerings(self):
    self.set_http_response(status_code=200)
    response = self.ec2.get_all_reserved_instances_offerings()
    self.assertEqual(len(response), 2)
    instance = response[0]
    self.assertEqual(instance.id, '2964d1bf71d8')
    self.assertEqual(instance.instance_type, 'c1.medium')
    self.assertEqual(instance.availability_zone, 'us-east-1c')
    self.assertEqual(instance.duration, 94608000)
    self.assertEqual(instance.fixed_price, '775.0')
    self.assertEqual(instance.usage_price, '0.0')
    self.assertEqual(instance.description, 'product description')
    self.assertEqual(instance.instance_tenancy, 'default')
    self.assertEqual(instance.currency_code, 'USD')
    self.assertEqual(instance.offering_type, 'Heavy Utilization')
    self.assertEqual(len(instance.recurring_charges), 1)
    self.assertEqual(instance.recurring_charges[0].frequency, 'Hourly')
    self.assertEqual(instance.recurring_charges[0].amount, '0.095')
    self.assertEqual(len(instance.pricing_details), 1)
    self.assertEqual(instance.pricing_details[0].price, '0.045')
    self.assertEqual(instance.pricing_details[0].count, '1')