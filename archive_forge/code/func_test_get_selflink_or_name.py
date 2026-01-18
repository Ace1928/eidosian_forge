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
def test_get_selflink_or_name(self):
    network = self.driver.ex_get_network('lcnetwork')
    actual = self.driver._get_selflink_or_name(network, False, 'network')
    self.assertEqual('lcnetwork', actual)
    actual = self.driver._get_selflink_or_name(network, True, 'network')
    self.assertTrue(actual.startswith('https://'))
    actual = self.driver._get_selflink_or_name('lcnetwork', True, 'network')
    self.assertTrue(actual.startswith('https://'))
    actual = self.driver._get_selflink_or_name('lcnetwork', False, 'network')
    self.assertTrue('lcnetwork', actual)
    self.assertRaises(ValueError, self.driver._get_selflink_or_name, 'lcnetwork', True)