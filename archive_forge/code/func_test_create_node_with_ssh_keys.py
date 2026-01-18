import re
import sys
import json
import base64
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.compute import providers
from libcloud.utils.py3 import httplib, ensure_string
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.test.secrets import UPCLOUD_PARAMS
from libcloud.compute.types import Provider, NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.upcloud import UpcloudDriver, UpcloudResponse
def test_create_node_with_ssh_keys(self):
    image = NodeImage(id='01000000-0000-4000-8000-000030060200', name='Ubuntu Server 16.04 LTS (Xenial Xerus)', extra={'type': 'template'}, driver=self.driver)
    location = NodeLocation(id='fi-hel1', name='Helsinki #1', country='FI', driver=self.driver)
    size = NodeSize(id='1xCPU-1GB', name='1xCPU-1GB', ram=1024, disk=30, bandwidth=2048, extra={'storage_tier': 'maxiops'}, price=None, driver=self.driver)
    auth = NodeAuthSSHKey('publikey')
    node = self.driver.create_node(name='test_server', size=size, image=image, location=location, auth=auth)
    self.assertTrue(re.match('^[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}$', node.id))
    self.assertEqual(node.name, 'test_server')
    self.assertEqual(node.state, NodeState.STARTING)
    self.assertTrue(len(node.public_ips) > 0)
    self.assertTrue(len(node.private_ips) > 0)
    self.assertEqual(node.driver, self.driver)