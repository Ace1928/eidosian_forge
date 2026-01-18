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
def test_ex_list_forwarding_rules(self):
    forwarding_rules = self.driver.ex_list_forwarding_rules()
    forwarding_rules_all = self.driver.ex_list_forwarding_rules('all')
    forwarding_rules_uc1 = self.driver.ex_list_forwarding_rules('us-central1')
    self.assertEqual(len(forwarding_rules), 2)
    self.assertEqual(len(forwarding_rules_all), 2)
    self.assertEqual(forwarding_rules[0].name, 'lcforwardingrule')
    self.assertEqual(forwarding_rules_uc1[0].name, 'lcforwardingrule')
    names = [f.name for f in forwarding_rules_all]
    self.assertTrue('lcforwardingrule' in names)