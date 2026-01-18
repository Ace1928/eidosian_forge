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
def test_ex_update_firewall(self):
    firewall_name = 'lcfirewall'
    firewall = self.driver.ex_get_firewall(firewall_name)
    firewall.source_ranges = ['10.0.0.0/16']
    firewall.description = 'LCFirewall-2'
    firewall2 = self.driver.ex_update_firewall(firewall)
    self.assertTrue(isinstance(firewall2, GCEFirewall))