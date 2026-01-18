import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_create_node_network_opts(self):
    node_name = 'node-name'
    size = self.driver.ex_get_size('n1-standard-1')
    image = self.driver.ex_get_image('debian-7')
    zone = self.driver.ex_get_zone('us-central1-a')
    network = self.driver.ex_get_network('lcnetwork')
    address = self.driver.ex_get_address('lcaddress')
    ex_nic_gce_struct = [{'network': 'global/networks/lcnetwork', 'accessConfigs': [{'name': 'lcnetwork-test', 'type': 'ONE_TO_ONE_NAT'}]}]
    node = self.driver.create_node(node_name, size, image)
    self.assertEqual(node.extra['networkInterfaces'][0]['name'], 'nic0')
    node = self.driver.create_node(node_name, size, image, location=zone, ex_network=network)
    self.assertEqual(node.extra['networkInterfaces'][0]['name'], 'nic0')
    node = self.driver.create_node(node_name, size, image, location=zone, ex_nic_gce_struct=ex_nic_gce_struct)
    self.assertEqual(node.extra['networkInterfaces'][0]['name'], 'nic0')
    self.assertRaises(ValueError, self.driver.create_node, node_name, size, image, location=zone, external_ip=address, ex_nic_gce_struct=ex_nic_gce_struct)
    self.assertRaises(ValueError, self.driver.create_node, node_name, size, image, location=zone, ex_network=network, ex_nic_gce_struct=ex_nic_gce_struct)