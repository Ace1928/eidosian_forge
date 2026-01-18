import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
def test_get_regions_overrides(self):
    ec2_regions = get_regions('ec2', region_cls=TestRegionInfo, connection_cls=FakeConn)
    self.assertTrue(len(ec2_regions) >= 10)
    west_2 = None
    for region_info in ec2_regions:
        if region_info.name == 'us-west-2':
            west_2 = region_info
            break
    self.assertNotEqual(west_2, None, "Couldn't find the us-west-2 region!")
    self.assertFalse(isinstance(west_2, RegionInfo))
    self.assertTrue(isinstance(west_2, TestRegionInfo))
    self.assertEqual(west_2.name, 'us-west-2')
    self.assertEqual(west_2.endpoint, 'ec2.us-west-2.amazonaws.com')
    self.assertEqual(west_2.connection_cls, FakeConn)