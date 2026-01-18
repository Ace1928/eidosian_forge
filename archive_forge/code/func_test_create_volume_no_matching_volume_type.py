import os
import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.common.types import ProviderError
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import (
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver, CloudStackAffinityGroupType
def test_create_volume_no_matching_volume_type(self):
    """If the ex_disk_type does not exit, then an exception should be
        thrown."""
    location = self.driver.list_locations()[0]
    self.assertRaises(LibcloudError, self.driver.create_volume, 'vol-0', location, 11, ex_volume_type='FooVolumeType')