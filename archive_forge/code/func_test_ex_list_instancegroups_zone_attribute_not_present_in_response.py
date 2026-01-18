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
def test_ex_list_instancegroups_zone_attribute_not_present_in_response(self):
    GCEMockHttp.type = 'zone_attribute_not_present'
    loc = 'us-central1-a'
    actual = self.driver.ex_list_instancegroups(loc)
    self.assertTrue(len(actual) == 2)
    self.assertEqual(actual[0].name, 'myname')
    self.assertEqual(actual[1].name, 'myname2')