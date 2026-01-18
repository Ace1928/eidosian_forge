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
def test_ex_create_route(self):
    route_name = 'lcdemoroute'
    dest_range = '192.168.25.0/24'
    priority = 1000
    route = self.driver.ex_create_route(route_name, dest_range)
    self.assertTrue(isinstance(route, GCERoute))
    self.assertEqual(route.name, route_name)
    self.assertEqual(route.priority, priority)
    self.assertTrue('tag1' in route.tags)
    self.assertTrue(route.extra['nextHopInstance'].endswith('libcloud-100'))
    self.assertEqual(route.dest_range, dest_range)