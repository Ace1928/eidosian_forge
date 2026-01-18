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
def test_build_service_account_gce_struct_custom_service_account(self):
    data = [{'email': '1', 'scopes': ['a']}, {'email': '2', 'scopes': ['b']}]
    expected_result = [{'email': '1', 'scopes': ['https://www.googleapis.com/auth/a']}, {'email': '2', 'scopes': ['https://www.googleapis.com/auth/b']}]
    result = self.driver._build_service_accounts_gce_list(service_accounts=data)
    self.assertEqual(result, expected_result)