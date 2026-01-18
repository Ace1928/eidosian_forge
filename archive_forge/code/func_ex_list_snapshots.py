import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def ex_list_snapshots(self):
    snapshots = self.driver.ex_list_snapshots()
    self.assertEqual(len(snapshots), 2)
    snapshot = snapshots[0]
    self.assertEqual(snapshot.id, '123')
    self.assertEqual(snapshot.size, '25.0')
    self.assertEqual(snapshot.state, 'available')
    snapshot = snapshots[1]
    self.assertEqual(snapshot.id, '1234')
    self.assertEqual(snapshot.size, '25.0')
    self.assertEqual(snapshot.state, 'creating')