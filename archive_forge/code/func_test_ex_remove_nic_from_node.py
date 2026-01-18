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
def test_ex_remove_nic_from_node(self):
    vm = self.driver.list_nodes()[0]
    nic = self.driver.ex_list_nics(node=vm)[0]
    result = self.driver.ex_detach_nic_from_node(node=vm, nic=nic)
    self.assertTrue(result)