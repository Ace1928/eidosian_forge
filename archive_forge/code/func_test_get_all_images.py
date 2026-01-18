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
def test_get_all_images(self):
    self.set_http_response(status_code=200)
    parsed = self.ec2.get_all_images()
    self.assertEquals(1, len(parsed))
    self.assertEquals('ami-abcd1234', parsed[0].id)
    self.assertEquals('111111111111/windows2008r2-hvm-i386-20130702', parsed[0].location)
    self.assertEquals('available', parsed[0].state)
    self.assertEquals('111111111111', parsed[0].ownerId)
    self.assertEquals('111111111111', parsed[0].owner_id)
    self.assertEquals(False, parsed[0].is_public)
    self.assertEquals('i386', parsed[0].architecture)
    self.assertEquals('machine', parsed[0].type)
    self.assertEquals(None, parsed[0].kernel_id)
    self.assertEquals(None, parsed[0].ramdisk_id)
    self.assertEquals(None, parsed[0].owner_alias)
    self.assertEquals('windows', parsed[0].platform)
    self.assertEquals('Windows Test', parsed[0].name)
    self.assertEquals('Windows Test Description', parsed[0].description)
    self.assertEquals('ebs', parsed[0].root_device_type)
    self.assertEquals('/dev/sda1', parsed[0].root_device_name)
    self.assertEquals('hvm', parsed[0].virtualization_type)
    self.assertEquals('xen', parsed[0].hypervisor)
    self.assertEquals(None, parsed[0].instance_lifecycle)
    self.assertEquals(1, len(parsed[0].billing_products))
    self.assertEquals('bp-6ba54002', parsed[0].billing_products[0])
    self.assertEquals(5, len(parsed[0].block_device_mapping))