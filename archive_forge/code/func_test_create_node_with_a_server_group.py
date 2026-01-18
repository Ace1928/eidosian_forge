import sys
import base64
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import BRIGHTBOX_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.brightbox import BrightboxNodeDriver
def test_create_node_with_a_server_group(self):
    size = self.driver.list_sizes()[0]
    image = self.driver.list_images()[0]
    node = self.driver.create_node(name='Test Node', image=image, size=size, ex_servergroup='grp-12345')
    self.assertEqual('gb1-a', node.extra['zone'].name)
    self.assertEqual(len(node.extra['server_groups']), 1)
    self.assertEqual(node.extra['server_groups'][0]['id'], 'grp-12345')