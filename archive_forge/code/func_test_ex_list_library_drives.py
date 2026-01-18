import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_list_library_drives(self):
    drives = self.driver.ex_list_library_drives()
    drive = drives[0]
    self.assertEqual(drive.name, 'IPCop 2.0.2')
    self.assertEqual(drive.size, 1)
    self.assertEqual(drive.media, 'cdrom')
    self.assertEqual(drive.status, 'unmounted')