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
def test_ex_list_targethttpproxies(self):
    target_proxies = self.driver.ex_list_targethttpproxies()
    self.assertEqual(len(target_proxies), 2)
    self.assertEqual(target_proxies[0].name, 'web-proxy')
    names = [t.name for t in target_proxies]
    self.assertListEqual(names, ['web-proxy', 'web-proxy2'])