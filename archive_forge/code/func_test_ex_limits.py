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
def test_ex_limits(self):
    limits = self.driver.ex_limits()
    self.assertEqual(limits['max_images'], 20)
    self.assertEqual(limits['max_networks'], 20)
    self.assertEqual(limits['max_public_ips'], -1)
    self.assertEqual(limits['max_vpc'], 20)
    self.assertEqual(limits['max_instances'], 20)
    self.assertEqual(limits['max_projects'], -1)
    self.assertEqual(limits['max_volumes'], 20)
    self.assertEqual(limits['max_snapshots'], 20)