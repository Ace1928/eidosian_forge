import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_attach_volume_invalid_usecase(self):
    node = self.driver.ex_get_node('node-name')
    self.assertRaises(ValueError, self.driver.attach_volume, node, None)
    self.assertRaises(ValueError, self.driver.attach_volume, node, None, ex_source='foo/bar', device=None)