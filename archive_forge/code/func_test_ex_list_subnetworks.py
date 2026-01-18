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
def test_ex_list_subnetworks(self):
    subnetworks = self.driver.ex_list_subnetworks()
    self.assertEqual(len(subnetworks), 1)
    self.assertEqual(subnetworks[0].name, 'cf-972cf02e6ad49112')
    self.assertEqual(subnetworks[0].cidr, '10.128.0.0/20')
    subnetworks = self.driver.ex_list_subnetworks('all')
    self.assertEqual(len(subnetworks), 4)