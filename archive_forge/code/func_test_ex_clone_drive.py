import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_clone_drive(self):
    drive = self.driver.ex_list_user_drives()[0]
    cloned_drive = self.driver.ex_clone_drive(drive=drive, name='cloned drive')
    self.assertEqual(cloned_drive.name, 'cloned drive')