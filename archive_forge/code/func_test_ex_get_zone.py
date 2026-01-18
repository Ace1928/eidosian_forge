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
def test_ex_get_zone(self):
    zone_name = 'us-central1-b'
    zone = self.driver.ex_get_zone(zone_name)
    self.assertEqual(zone.name, zone_name)
    self.assertFalse(zone.time_until_mw)
    self.assertFalse(zone.next_mw_duration)
    zone_no_mw = self.driver.ex_get_zone('us-central1-a')
    self.assertIsNone(zone_no_mw.time_until_mw)