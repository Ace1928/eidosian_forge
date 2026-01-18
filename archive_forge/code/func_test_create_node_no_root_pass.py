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
def test_create_node_no_root_pass(self):
    image = self.driver.list_images()[0]
    size = self.driver.list_sizes()[0]
    location = self.driver.list_locations()[0]
    with self.assertRaises(LinodeExceptionV4):
        self.driver.create_node(location, size, image=image, name='TestNode')