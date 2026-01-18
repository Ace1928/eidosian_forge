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
def test_ex_list_addresses(self):
    address_list = self.driver.ex_list_addresses()
    address_list_all = self.driver.ex_list_addresses('all')
    address_list_uc1 = self.driver.ex_list_addresses('us-central1')
    address_list_global = self.driver.ex_list_addresses('global')
    self.assertEqual(len(address_list), 2)
    self.assertEqual(len(address_list_all), 5)
    self.assertEqual(len(address_list_global), 1)
    self.assertEqual(address_list[0].name, 'libcloud-demo-address')
    self.assertEqual(address_list_uc1[0].name, 'libcloud-demo-address')
    self.assertEqual(address_list_global[0].name, 'lcaddressglobal')
    names = [a.name for a in address_list_all]
    self.assertTrue('libcloud-demo-address' in names)