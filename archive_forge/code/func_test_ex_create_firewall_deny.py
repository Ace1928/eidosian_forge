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
def test_ex_create_firewall_deny(self):
    name = 'lcfirewall-deny'
    priority = 900
    denied = [{'IPProtocol': 'tcp', 'ports': ['4567']}]
    description = 'Libcloud Deny Firewall'
    source_ranges = ['10.240.100.0/24']
    source_tags = ['libcloud']
    network = 'default'
    firewall = self.driver.ex_create_firewall(name, denied=denied, description=description, network=network, priority=priority, source_tags=source_tags, source_ranges=source_ranges)
    self.assertTrue(isinstance(firewall, GCEFirewall))
    self.assertEqual(firewall.name, name)