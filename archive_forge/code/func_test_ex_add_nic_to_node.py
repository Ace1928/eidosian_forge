import os
import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.common.types import ProviderError
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import (
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver, CloudStackAffinityGroupType
def test_ex_add_nic_to_node(self):
    vm = self.driver.list_nodes()[0]
    network = self.driver.ex_list_networks()[0]
    ip = '10.1.4.123'
    result = self.driver.ex_attach_nic_to_node(node=vm, network=network, ip_address=ip)
    self.assertTrue(result)