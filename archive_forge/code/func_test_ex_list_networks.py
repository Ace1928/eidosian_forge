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
def test_ex_list_networks(self):
    networks = self.driver.ex_list_networks()
    self.assertEqual(len(networks), 3)
    self.assertEqual(networks[0].name, 'cf')
    self.assertEqual(networks[0].mode, 'auto')
    self.assertEqual(len(networks[0].subnetworks), 4)
    self.assertEqual(networks[1].name, 'custom')
    self.assertEqual(networks[1].mode, 'custom')
    self.assertEqual(len(networks[1].subnetworks), 1)
    self.assertEqual(networks[2].name, 'default')
    self.assertEqual(networks[2].mode, 'legacy')