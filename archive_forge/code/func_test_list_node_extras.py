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
def test_list_node_extras(self):
    nodes = self.driver.list_nodes()
    self.assertFalse(nodes[0].size is None)
    self.assertFalse(nodes[1].size is None)
    self.assertFalse(nodes[0].image is None)
    self.assertFalse(nodes[1].image is None)
    self.assertEqual(nodes[0].image.id, 'img-arm8f')
    self.assertEqual(nodes[0].size.id, 'typ-urtky')
    self.assertEqual(nodes[1].image.id, 'img-j93gd')
    self.assertEqual(nodes[1].size.id, 'typ-qdiwq')
    self.assertEqual(nodes[0].extra['fqdn'], 'srv-xvpn7.gb1.brightbox.com')
    self.assertEqual(nodes[1].extra['fqdn'], 'srv-742vn.gb1.brightbox.com')
    self.assertEqual(nodes[0].extra['hostname'], 'srv-xvpn7')
    self.assertEqual(nodes[1].extra['hostname'], 'srv-742vn')
    self.assertEqual(nodes[0].extra['status'], 'active')
    self.assertEqual(nodes[1].extra['status'], 'active')
    self.assertTrue('interfaces' in nodes[0].extra)
    self.assertTrue('zone' in nodes[0].extra)
    self.assertTrue('snapshots' in nodes[0].extra)
    self.assertTrue('server_groups' in nodes[0].extra)
    self.assertTrue('started_at' in nodes[0].extra)
    self.assertTrue('created_at' in nodes[0].extra)
    self.assertFalse('deleted_at' in nodes[0].extra)