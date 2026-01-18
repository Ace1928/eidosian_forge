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
def test_ex_get_route(self):
    route_name = 'lcdemoroute'
    route = self.driver.ex_get_route(route_name)
    self.assertEqual(route.name, route_name)
    self.assertEqual(route.dest_range, '192.168.25.0/24')
    self.assertEqual(route.priority, 1000)