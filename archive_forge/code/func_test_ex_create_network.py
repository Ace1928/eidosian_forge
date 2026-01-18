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
def test_ex_create_network(self):
    network_name = 'lcnetwork'
    cidr = '10.11.0.0/16'
    routing_mode = 'REGIONAL'
    network = self.driver.ex_create_network(network_name, cidr, routing_mode='regional')
    self.assertTrue(isinstance(network, GCENetwork))
    self.assertEqual(network.name, network_name)
    self.assertEqual(network.cidr, cidr)
    description = 'A custom network'
    network = self.driver.ex_create_network(network_name, cidr, description=description, routing_mode=routing_mode)
    self.assertEqual(network.extra['description'], description)
    self.assertEqual(network.extra['routingConfig']['routingMode'], routing_mode)