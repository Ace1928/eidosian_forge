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
def test_list_disktypes(self):
    disktypes = self.driver.ex_list_disktypes()
    disktypes_all = self.driver.ex_list_disktypes('all')
    disktypes_uc1a = self.driver.ex_list_disktypes('us-central1-a')
    self.assertEqual(len(disktypes), 2)
    self.assertEqual(len(disktypes_all), 9)
    self.assertEqual(len(disktypes_uc1a), 2)
    self.assertEqual(disktypes[0].name, 'pd-ssd')
    self.assertEqual(disktypes_uc1a[0].name, 'pd-ssd')
    names = [v.name for v in disktypes_all]
    self.assertTrue('pd-standard' in names)
    self.assertTrue('local-ssd' in names)