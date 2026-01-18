import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume, NodeAuthSSHKey, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.linode import LinodeNodeDriver
def test_create_node_response(self):
    node = self.driver.create_node(name='node-name', location=self.driver.list_locations()[0], size=self.driver.list_sizes()[0], image=self.driver.list_images()[0], auth=NodeAuthPassword('foobar'))
    self.assertTrue(isinstance(node, Node))