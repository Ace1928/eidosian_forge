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
def test_ex_create_targethttpproxy(self):
    proxy_name = 'web-proxy'
    urlmap_name = 'web-map'
    for urlmap in (urlmap_name, self.driver.ex_get_urlmap(urlmap_name)):
        proxy = self.driver.ex_create_targethttpproxy(proxy_name, urlmap)
        self.assertTrue(isinstance(proxy, GCETargetHttpProxy))
        self.assertEqual(proxy_name, proxy.name)