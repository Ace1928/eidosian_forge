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
def test_ex_create_forwarding_rule_targetpool_keyword(self):
    """Test backwards compatibility with the targetpool kwarg."""
    fwr_name = 'lcforwardingrule'
    targetpool = 'lctargetpool'
    region = 'us-central1'
    address = self.driver.ex_get_address('lcaddress')
    port_range = '8000-8500'
    description = 'test forwarding rule'
    fwr = self.driver.ex_create_forwarding_rule(fwr_name, targetpool=targetpool, region=region, address=address, port_range=port_range, description=description)
    self.assertTrue(isinstance(fwr, GCEForwardingRule))
    self.assertEqual(fwr.name, fwr_name)
    self.assertEqual(fwr.region.name, region)
    self.assertEqual(fwr.protocol, 'TCP')
    self.assertEqual(fwr.extra['portRange'], port_range)
    self.assertEqual(fwr.extra['description'], description)