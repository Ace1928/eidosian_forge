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
def test_ex_create_snapshot_template(self):
    snapshot = self.driver.list_snapshots()[0]
    template = self.driver.ex_create_snapshot_template(snapshot, 'test-libcloud-template', 99)
    self.assertEqual(template.id, '10260')
    self.assertEqual(template.name, 'test-libcloud-template')
    self.assertEqual(template.extra['displaytext'], 'test-libcloud-template')
    self.assertEqual(template.extra['hypervisor'], 'VMware')
    self.assertEqual(template.extra['os'], 'Other Linux (64-bit)')