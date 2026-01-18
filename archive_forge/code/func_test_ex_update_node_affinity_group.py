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
def test_ex_update_node_affinity_group(self):
    affinity_group_list = self.driver.ex_list_affinity_groups()
    nodes = self.driver.list_nodes()
    node = self.driver.ex_update_node_affinity_group(nodes[0], affinity_group_list)
    self.assertEqual(node.extra['affinity_group'][0], affinity_group_list[0].id)