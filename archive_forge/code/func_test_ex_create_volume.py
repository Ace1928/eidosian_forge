import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume, NodeAuthSSHKey, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.linode import LinodeNodeDriver
def test_ex_create_volume(self):
    node = self.driver.list_nodes()[0]
    volume = self.driver.ex_create_volume(size=4096, name='Another test image', node=node, fs_type='ext4')
    self.assertTrue(isinstance(volume, StorageVolume))