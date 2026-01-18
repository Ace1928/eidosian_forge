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
def test_ex_create_network_bad_options(self):
    network_name = 'lcnetwork'
    cidr = '10.11.0.0/16'
    self.assertRaises(ValueError, self.driver.ex_create_network, network_name, cidr, mode='auto')
    self.assertRaises(ValueError, self.driver.ex_create_network, network_name, cidr, mode='foobar')
    self.assertRaises(ValueError, self.driver.ex_create_network, network_name, None, mode='legacy')
    self.assertRaises(ValueError, self.driver.ex_create_network, network_name, cidr, routing_mode='universal')