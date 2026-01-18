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
def test_create_node_subnetwork_opts(self):
    node_name = 'sn-node-name'
    size = self.driver.ex_get_size('n1-standard-1')
    image = self.driver.ex_get_image('debian-7')
    zone = self.driver.ex_get_zone('us-central1-a')
    network = self.driver.ex_get_network('custom-network')
    subnetwork = self.driver.ex_get_subnetwork('cf-972cf02e6ad49112')
    ex_nic_gce_struct = [{'network': 'global/networks/custom-network', 'subnetwork': 'projects/project_name/regions/us-central1/subnetworks/cf-972cf02e6ad49112', 'accessConfigs': [{'name': 'External NAT', 'type': 'ONE_TO_ONE_NAT'}]}]
    node = self.driver.create_node(node_name, size, image, location=zone, ex_network=network, ex_subnetwork=subnetwork)
    self.assertEqual(node.extra['networkInterfaces'][0]['name'], 'nic0')
    self.assertEqual(node.extra['networkInterfaces'][0]['subnetwork'].split('/')[-1], 'cf-972cf02e6ad49112')
    node = self.driver.create_node(node_name, size, image, location=zone, ex_nic_gce_struct=ex_nic_gce_struct)
    self.assertEqual(node.extra['networkInterfaces'][0]['name'], 'nic0')
    self.assertEqual(node.extra['networkInterfaces'][0]['subnetwork'].split('/')[-1], 'cf-972cf02e6ad49112')
    node = self.driver.create_node(node_name, size, image, location=zone, ex_network=network, ex_subnetwork=subnetwork.extra['selfLink'])
    self.assertEqual(node.extra['networkInterfaces'][0]['name'], 'nic0')
    self.assertEqual(node.extra['networkInterfaces'][0]['subnetwork'].split('/')[-1], 'cf-972cf02e6ad49112')