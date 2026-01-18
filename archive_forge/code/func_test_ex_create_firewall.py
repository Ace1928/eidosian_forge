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
def test_ex_create_firewall(self):
    name = 'lcfirewall'
    priority = 900
    description = 'Libcloud Test Firewall'
    allowed = [{'IPProtocol': 'tcp', 'ports': ['4567']}]
    source_service_accounts = ['lcsource@gserviceaccount.com']
    target_tags = ['libcloud']
    network = 'default'
    firewall = self.driver.ex_create_firewall(name, allowed, description=description, network=network, priority=priority, target_tags=target_tags, source_service_accounts=source_service_accounts)
    self.assertTrue(isinstance(firewall, GCEFirewall))
    self.assertEqual(firewall.name, name)