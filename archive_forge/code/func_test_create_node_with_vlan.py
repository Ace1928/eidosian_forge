import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_create_node_with_vlan(self):
    image = self.driver.list_images()[0]
    size = self.driver.list_sizes()[0]
    vlan_uuid = '39ae851d-433f-4ac2-a803-ffa24cb1fa3e'
    node = self.driver.create_node(name='test node vlan', size=size, image=image, ex_vlan=vlan_uuid)
    self.assertEqual(node.name, 'test node vlan')
    self.assertEqual(len(node.extra['nics']), 2)
    self.assertEqual(node.extra['nics'][0]['ip_v4_conf']['conf'], 'dhcp')
    self.assertEqual(node.extra['nics'][1]['vlan']['uuid'], vlan_uuid)