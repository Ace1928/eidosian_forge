import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_tag_resource_invalid_resource_object(self):
    tag = self.driver.ex_list_tags()[0]
    expected_msg = "Resource doesn't have id attribute"
    assertRaisesRegex(self, ValueError, expected_msg, self.driver.ex_tag_resource, tag=tag, resource={})