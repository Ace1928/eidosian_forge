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
def test_ex_destroy_disk(self):
    node = Node('22344420', None, NodeState.STOPPED, None, None, driver=self.driver)
    disk = LinodeDisk('23517413', None, None, None, self.driver, None)
    result = self.driver.ex_destroy_disk(node, disk)
    self.assertTrue(result)