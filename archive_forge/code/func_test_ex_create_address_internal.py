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
def test_ex_create_address_internal(self):
    address_name = 'lcaddressinternal'
    address = self.driver.ex_create_address(address_name, region='us-central1', address='10.128.0.12', address_type='INTERNAL', subnetwork='subnet-1')
    self.assertTrue(isinstance(address, GCEAddress))
    self.assertEqual(address.name, address_name)
    self.assertEqual(address.address, '10.128.0.12')
    self.assertRaises(ValueError, self.driver.ex_create_address, address_name, address_type='WRONG')
    self.assertRaises(ValueError, self.driver.ex_create_address, address_name, address_type='EXTERNAL', subnetwork='subnet-1')