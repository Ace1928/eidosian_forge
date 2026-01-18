import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_get_tag(self):
    tag = self.driver.ex_get_tag(tag_id='a010ec41-2ead-4630-a1d0-237fa77e4d4d')
    self.assertEqual(tag.id, 'a010ec41-2ead-4630-a1d0-237fa77e4d4d')
    self.assertEqual(tag.name, 'test tag 2')