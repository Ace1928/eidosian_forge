import sys
import random
import string
import unittest
from libcloud.utils.py3 import httplib
from libcloud.common.gandi import GandiException
from libcloud.test.secrets import GANDI_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gandi import GandiNodeDriver
from libcloud.test.common.test_gandi import BaseGandiMockHttp
def test_ex_snapshot_disk(self):
    disks = self.driver.list_volumes()
    self.assertTrue(self.driver.ex_snapshot_disk(disks[2]))
    self.assertRaises(GandiException, self.driver.ex_snapshot_disk, disks[0])