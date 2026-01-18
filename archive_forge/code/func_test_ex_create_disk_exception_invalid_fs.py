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
def test_ex_create_disk_exception_invalid_fs(self):
    node = Node('22344420', None, None, None, None, driver=self.driver)
    image = self.driver.list_images()[0]
    with self.assertRaises(LinodeExceptionV4):
        self.driver.ex_create_disk(5000, 'TestDisk', node, 'random_fs', image=image, ex_root_pass='testing123')