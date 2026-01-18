import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeImage, NodeState, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.common.linode import LinodeDisk, LinodeIPAddress, LinodeExceptionV4
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.linode import LinodeNodeDriver, LinodeNodeDriverV4
def test_create_volume_unattached(self):
    location = self.driver.list_locations()[0]
    LinodeMockHttpV4.type = 'UNATTACHED'
    volume = self.driver.create_volume('Volume1', 50, location=location, tags=['test123', 'testing'])
    self.assertEqual(volume.size, 50)
    self.assertEqual(volume.name, 'Volume1')
    self.assertEqual(volume.extra['tags'], ['test123', 'testing'])