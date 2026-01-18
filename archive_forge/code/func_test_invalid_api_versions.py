import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_invalid_api_versions(self):
    expected_msg = 'Unsupported API version: invalid'
    assertRaisesRegex(self, NotImplementedError, expected_msg, CloudSigmaNodeDriver, 'username', 'password', api_version='invalid')