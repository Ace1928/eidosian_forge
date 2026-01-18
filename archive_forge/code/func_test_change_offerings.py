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
def test_change_offerings(self):
    offering = NodeSize('eee-fff-ggg-hhh', 'fake-size', 1, 4, 5, 0.1, None)
    node = self.driver.list_nodes()[0]
    res = node.ex_change_node_size(offering=offering)
    self.assertEqual(res, offering.id)