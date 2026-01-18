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
def test_ex_list_urlmaps(self):
    urlmaps_list = self.driver.ex_list_urlmaps()
    web_map = urlmaps_list[0]
    self.assertEqual(web_map.name, 'web-map')
    self.assertEqual(len(web_map.host_rules), 0)
    self.assertEqual(len(web_map.path_matchers), 0)
    self.assertEqual(len(web_map.tests), 0)