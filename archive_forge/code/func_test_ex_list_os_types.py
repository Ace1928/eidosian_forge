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
def test_ex_list_os_types(self):
    os_types = self.driver.ex_list_os_types()
    self.assertEqual(len(os_types), 146)
    self.assertEqual(os_types[0]['id'], 69)
    self.assertEqual(os_types[0]['oscategoryid'], 7)
    self.assertEqual(os_types[0]['description'], 'Asianux 3(32-bit)')