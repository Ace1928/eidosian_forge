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
def test_build_service_account_gce_list(self):
    self.assertRaises(ValueError, self.driver._build_service_accounts_gce_list, 'foo')
    actual = self.driver._build_service_accounts_gce_list()
    self.assertTrue(len(actual) == 1)
    self.assertTrue('email' in actual[0])
    self.assertTrue('scopes' in actual[0])