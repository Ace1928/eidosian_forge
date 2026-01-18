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
def test_ex_get_address(self):
    address_name = 'lcaddress'
    address = self.driver.ex_get_address(address_name)
    self.assertEqual(address.name, address_name)
    self.assertEqual(address.address, '173.255.113.20')
    self.assertEqual(address.region.name, 'us-central1')
    self.assertEqual(address.extra['status'], 'RESERVED')